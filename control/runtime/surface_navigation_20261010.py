"""水面定点导航的纯计算部分：UWB解帧、定位、IMU零点/滤波、航向PI。

坐标：A0=(0,0)、A1=(L,0)、A2=(L,W)、A3=(0,W)，+y为初始鱼头左侧。
距离单位为米，角度为度，时间为秒；复用现有 action1 的中位相对角定义。
"""
from __future__ import annotations

import math
import re
import statistics
from collections import deque
from dataclasses import dataclass, fields, replace

from .rl_actions_20260914 import TailAction, RLActionTrajectory, validate_trajectory
from .rl_action_scheduler_20260914 import RLActionScheduler, ScheduledRLAction
from .rl_sensor_calibration_20260914 import finite_float, wrap180, circular_mean_deg


@dataclass(frozen=True)
class NavigationConfig:
    L: float
    W: float
    x_d: float
    y_d: float
    theta_max: float = 30.
    theta_0: float = 10.
    theta_00: float = 10.
    t_0: float = .3
    t1: float = 15.
    t2: float = 5.
    t: float | None = None  # 总运动时长；None不设时间上限，不含放置/标定。
    kp: float = 1.
    ki: float = 0.
    yaw_error_scale_deg: float = 180.
    delta_yaw: float = 10.  # 兼容旧配置；不再参与到点/保持判定。
    delta_s: float = .2
    imu_filter_tau_s: float = .15
    yaw_sign: float = 1.
    calibration_max_std_deg: float = 3.
    calibration_min_samples: int = 20
    imu_timeout_s: float = .5
    uwb_timeout_s: float = 2.
    boundary_tolerance_m: float = .25  # 兼容旧配置；已取消定位越界判断和坐标裁剪。
    range_residual_limit_m: float = .5  # 兼容旧配置；已取消测距残差检查。
    uwb_tag: str = 't0:0'
    H: float | None = None  # 基站天线到水面标签天线的垂直距离；None兼容旧的未知高度配置。

    def __post_init__(self):
        for item in fields(self):
            if item.name in ('t', 'H') and getattr(self, item.name) is None:
                continue
            if item.name != 'uwb_tag':
                object.__setattr__(self, item.name, finite_float(getattr(self, item.name), item.name))
        if self.t is not None and self.t <= 0:
            raise ValueError('t必须为正有限秒数，或null（不限时）')
        if self.H is not None and (self.H < 0 or not math.isfinite(self.H*self.H)):
            raise ValueError('H必须为非负且平方有限的高度，单位米')
        for name in ('L', 'W', 't2', 'yaw_error_scale_deg', 'delta_yaw', 'delta_s',
                     'calibration_max_std_deg', 'imu_timeout_s', 'uwb_timeout_s', 'range_residual_limit_m'):
            if getattr(self, name) <= 0:
                raise ValueError(f'{name} 必须大于0')
        for name in ('t_0', 't1', 'kp', 'ki', 'imu_filter_tau_s', 'boundary_tolerance_m'):
            if getattr(self, name) < 0:
                raise ValueError(f'{name} 不能为负')
        if not (0 <= self.theta_0 < self.theta_max <= 30 and 0 <= self.theta_00 < self.theta_max):
            raise ValueError('必须满足 30>=theta_max>theta_0>=0 且 theta_max>theta_00>=0')
        if not (0 <= self.x_d <= self.L and 0 <= self.y_d <= self.W):
            raise ValueError('目标点必须在 [0,L] × [0,W] 内')
        if self.yaw_sign not in (-1., 1.) or not 0 < self.delta_yaw <= 180:
            raise ValueError('yaw_sign须为±1，delta_yaw须在(0,180]内')
        if self.yaw_error_scale_deg > 180:
            raise ValueError('最短角差的归一化尺度须在(0,180]内')
        if self.calibration_min_samples < 2 or not self.calibration_min_samples.is_integer():
            raise ValueError('calibration_min_samples须为至少2的整数')
        object.__setattr__(self, 'calibration_min_samples', int(self.calibration_min_samples))
        if not isinstance(self.uwb_tag, str) or not re.fullmatch(r't\d+:\d+', self.uwb_tag):
            raise ValueError('uwb_tag格式须如t0:0')
        # 拒绝无法表示为纳秒的持续时间；t_0=0是显式的无插值动作。
        for name in ('t_0', 't1', 't2', 'imu_timeout_s', 'uwb_timeout_s'):
            value = getattr(self, name)
            if not math.isfinite(value*1e9) or (value > 0 and round(value*1e9) < 1):
                raise ValueError(f'{name}无法表示为有效纳秒数')

    @classmethod
    def from_robot(cls, robot, **overrides):
        values = dict(robot.get('surface_navigation_20261010', {}) or {})
        values.update({k: v for k, v in overrides.items() if v is not None})
        for name in ('L', 'W', 'x_d', 'y_d'):
            if values.get(name) is None:
                raise ValueError(f'请在robot.yaml的surface_navigation_20261010中填写{name}，或通过命令行指定')
        return cls(**values)


def validate_waypoints(cfg, waypoints):
    """规范化顺序点表；所有坐标和时长必须在启动采集/PWM前完成校验。"""
    if waypoints is None:
        return None
    if not waypoints:
        raise ValueError('waypoints至少需要一个目标点')
    result = []
    total = 0.
    for index, row in enumerate(waypoints, 1):
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            raise ValueError(f'第{index}个目标点必须为 X Y 秒数')
        x, y, duration = (finite_float(v, f'waypoint_{index}') for v in row)
        replace(cfg, x_d=x, y_d=y, t=duration)  # 复用单点的边界与正时长检查。
        if not math.isfinite(duration*1e9) or round(duration*1e9) < 1:
            raise ValueError(f'第{index}个目标点时长无法表示为有效纳秒数')
        previous = total
        total += duration
        if not math.isfinite(total*1e9) or total <= previous:
            raise ValueError('目标点累计时长过大或无法区分相邻时间边界')
        result.append((x, y, duration))
    return tuple(result)


@dataclass(frozen=True)
class UWBFrame:
    ranges_m: tuple[float | None, float | None, float | None, float | None]
    frame_id: str


class UWBStreamDecoder:
    """只消费mc帧；KT0是同一测量的低精度显示，不能重复触发动作。

必须传入raw_hex解出的原始字节，不能使用已strip的raw_text拼接串口片段。
实测mc 00仍有四路有效距离，因此不把第二字段当作有效位掩码。
"""
    def __init__(self, tag='t0:0'):
        self.tag = tag
        self._buffer = b''
        self._seen = deque(maxlen=64)
        self.invalid_frames = 0
        self.incomplete_frames = 0

    def reset_partial(self):
        self._buffer = b''

    def feed(self, chunk: bytes, *, include_incomplete=False) -> list[UWBFrame]:
        """默认只返回完整测距；导航可显式接收含None的缺基站报文以延用旧定位。"""
        self._buffer += chunk
        lines = self._buffer.split(b'\n')
        self._buffer = lines.pop()
        if len(self._buffer) > 4096:
            self._buffer = b''
            self.invalid_frames += 1
        result = []
        for line in lines:
            parts = line.strip().split()
            if not parts or parts[0] != b'mc':
                continue
            try:
                # 支持四路/八路mc距离格式，尾部通过tN:N标签定位计数器。
                tag_index = next(i for i, v in enumerate(parts) if re.fullmatch(rb't\d+:\d+', v))
                if tag_index not in (9, 13) or len(parts) != tag_index+2:
                    raise ValueError('mc字段数量无效')
                if parts[tag_index].decode('ascii') != self.tag:
                    continue
                if not all(re.fullmatch(rb'[0-9a-fA-F]{8}', v) for v in parts[2:6]):
                    raise ValueError('距离须为8位十六进制')
                distances = tuple(int(v, 16) for v in parts[2:6])
                identity = b':'.join(parts[tag_index-2:tag_index+1]).decode('ascii')
                if any(d <= 0 or d >= 0x7fffffff for d in distances):
                    self.invalid_frames += 1
                    self.incomplete_frames += 1
                    if include_incomplete:
                        result.append(UWBFrame(tuple(d/1000. if 0 < d < 0x7fffffff else None
                                                     for d in distances), identity))
                    continue  # 不占用完整帧去重ID，允许随后收到同ID的完整测距。
                if identity in self._seen:
                    continue
                self._seen.append(identity)
                result.append(UWBFrame(tuple(d/1000. for d in distances), identity))
            except (ValueError, StopIteration, UnicodeError):
                self.invalid_frames += 1
        return result


def position_from_ranges(ranges_m, L, W, boundary_tolerance_m=.25, range_residual_limit_m=.5, *, H=None):
    """四个同高基站平方差的对称解；公共垂直距离在差分中消去。

保留原始计算坐标，即使落在矩形外也不拒绝或裁剪；尚未增加位置滤波。
boundary_tolerance_m和range_residual_limit_m仅兼容旧调用，不参与定位判断。
"""
    L, W = finite_float(L, 'L'), finite_float(W, 'W')
    tolerance = finite_float(boundary_tolerance_m, 'boundary_tolerance_m')
    residual_limit = finite_float(range_residual_limit_m, 'range_residual_limit_m')
    distances = tuple(finite_float(r, 'range_m') for r in ranges_m)
    if L <= 0 or W <= 0 or tolerance < 0 or residual_limit <= 0 or len(distances) != 4 or min(distances) < 0:
        raise ValueError('四路距离/矩形尺寸无效')
    height_sq = None
    if H is not None:
        H = finite_float(H, 'H')
        if H < 0 or not math.isfinite(H*H):
            raise ValueError('基站与标签高度差H无效')
        if min(distances) < H:
            raise ValueError(f'测得斜距小于垂直高度H={H:g}m，请检查测距和天线高度')
        height_sq = H*H
    # 斜距平方减H²得到水平距离平方，无须开方；共同H²在后续差分中抵消。
    a, b, c, d = (r*r-(height_sq if height_sq is not None else 0.) for r in distances)
    x = (a-b+d-c+2*L*L)/(4*L)
    y = (a-d+b-c+2*W*W)/(4*W)
    # UWB噪声可能使实际位于池内的鱼算出池外坐标；不再根据矩形边界筛除。
    if not (math.isfinite(x) and math.isfinite(y)):
        raise ValueError('定位坐标不是有限数，请检查距离及L/W')
    # 直接采用平方差计算解，不再反算四路距离或根据残差拒绝定位。
    return x, y


class CalibratedIMU:
    """yaw/roll圆周零点与低通；pitch是[-90,90]读数的线性零点偏差。

pitch归零后的差值不强行夹到±90°，否则会隐藏相对于初始姿态的真实变化。
航向约定为从+x朝+y转动为正，安装方向不同可配置yaw_sign=-1。
"""
    def __init__(self, tau_s=.15, min_samples=20, max_std_deg=3., yaw_sign=1.):
        self.tau_s = finite_float(tau_s, 'imu_filter_tau_s')
        self.min_samples = int(min_samples)
        self.max_std_deg = finite_float(max_std_deg, 'calibration_max_std_deg')
        self.yaw_sign = finite_float(yaw_sign, 'yaw_sign')
        if self.tau_s < 0 or self.min_samples < 2 or self.max_std_deg <= 0 or self.yaw_sign not in (-1, 1):
            raise ValueError('IMU滤波/标定参数无效')
        self.zeros = None
        self.values = None
        self.last_t_ns = None
        self.diagnostics = {}

    @staticmethod
    def read_angles(data):
        result = {key: finite_float(data[key], key) for key in ('yaw_deg', 'pitch_deg', 'roll_deg')}
        if not -90 <= result['pitch_deg'] <= 90:
            raise ValueError('原始pitch不在[-90,90]内')
        return result

    def calibrate(self, rows):
        rows = [self.read_angles(row) for row in rows]
        if len(rows) < self.min_samples:
            raise ValueError(f'IMU标定样本不足：{len(rows)}/{self.min_samples}')
        zeros, stds = {}, {}
        for key in ('yaw_deg', 'pitch_deg', 'roll_deg'):
            values = [row[key] for row in rows]
            circular = key != 'pitch_deg'
            zero = circular_mean_deg(values) if circular else statistics.fmean(values)
            errors = [wrap180(v-zero) if circular else v-zero for v in values]
            stds[key] = math.sqrt(statistics.fmean(e*e for e in errors))
            if stds[key] > self.max_std_deg:
                raise ValueError(f'{key}标定期间不稳定：std={stds[key]:.3f}°')
            zeros[key] = zero
        self.zeros, self.values, self.last_t_ns = zeros, None, None
        self.diagnostics = {'zeros_deg': zeros, 'std_deg': stds, 'sample_count': len(rows)}
        return dict(self.diagnostics)

    def update(self, data, t_ns):
        if self.zeros is None:
            raise ValueError('IMU尚未标定')
        if self.last_t_ns is not None and t_ns <= self.last_t_ns:
            return dict(self.values)
        raw = self.read_angles(data)
        values = {key: (raw[key]-zero if key == 'pitch_deg' else wrap180(raw[key]-zero))
                  for key, zero in self.zeros.items()}
        values['yaw_deg'] = wrap180(self.yaw_sign*values['yaw_deg'])
        if self.values is not None and self.tau_s > 0:
            alpha = -math.expm1(-(t_ns-self.last_t_ns)/1e9/self.tau_s)
            for key, value in values.items():
                previous = self.values[key]
                delta = value-previous if key == 'pitch_deg' else wrap180(value-previous)
                filtered = previous + alpha*delta
                values[key] = filtered if key == 'pitch_deg' else wrap180(filtered)
        self.values, self.last_t_ns = values, int(t_ns)
        return dict(values)


@dataclass(frozen=True)
class NavigationDecision:
    x: float
    y: float
    yaw_deg: float
    yaw_d_deg: float
    yaw_error_deg: float
    distance_m: float
    effort: float
    theta_left: float
    theta_right: float
    reached: bool
    limited: bool
    pi_output: float = 0.  # 带符号的PI输出e；effort=abs(e)，供幅值计算和日志核对。
    control_mode: str = 'pi'
    # theta_left/theta_right保留日志字段名；强转向时表示第一/第二动作，可同号。


class NavigationPolicy:
    def __init__(self, config, calibration):
        self.config = config
        self.integral = 0.
        self.last_t_ns = None
        self.last_direction = 0
        centers, limits = calibration.centers, calibration.limits
        for sid in (1, 2, 3):
            lo, hi = limits[sid]
            if not all(math.isfinite(v) for v in (lo, hi, centers[sid])) or not lo <= centers[sid] <= hi:
                raise ValueError(f'尾鳍舵机{sid}中位/限位无效')
        # 共用theta取三舵机可用行程的交集，保持三个舵机同偏移联动。
        self.lower = max(-config.theta_max, *(limits[i][0]-centers[i] for i in (1, 2, 3)))
        self.upper = min(config.theta_max, *(limits[i][1]-centers[i] for i in (1, 2, 3)))

    def decide(self, x, y, yaw, t_ns):
        c = self.config
        x, y, yaw = finite_float(x, 'x'), finite_float(y, 'y'), wrap180(yaw)
        dx, dy = c.x_d-x, c.y_d-y
        distance = math.hypot(dx, dy)
        # 重合点没有几何方位；以当前航向定义期望，避免atan2(0,0)造成继续游走。
        desired = math.degrees(math.atan2(dy, dx)) if distance > 1e-9 else yaw
        error = wrap180(desired-yaw)
        direction = (error > 0) - (error < 0)
        # 到点仍优先暂停；大角差时绕过PI，按同侧30°、20°依次往复。
        reached = distance < c.delta_s
        if not reached and abs(error) >= 45.:
            self.integral = 0.
            self.last_t_ns = None  # 恢复PI时不把强转向所耗时间计入积分。
            self.last_direction = 0
            first, second = direction*30., direction*20.
            limited_first = min(self.upper, max(self.lower, first))
            limited_second = min(self.upper, max(self.lower, second))
            return NavigationDecision(x, y, yaw, desired, error, distance, 0.,
                                      limited_first, limited_second, False,
                                      limited_first != first or limited_second != second,
                                      control_mode='turn_positive' if direction > 0 else 'turn_negative')
        # yaw最短角差为[-180,180)，按可调角度尺度归一化并保留方向。
        normalized = min(1., max(-1., error/c.yaw_error_scale_deg))
        p = c.kp*normalized
        is_new = self.last_t_ns is None or t_ns > self.last_t_ns
        if is_new:
            dt = 0. if self.last_t_ns is None else (t_ns-self.last_t_ns)/1e9
            # 按用户的当前误差方向转向；换向先清积分，防止历史积分驱动错误方向。
            if direction != self.last_direction or direction == 0 or c.ki == 0:
                self.integral = 0.
            elif 0 < dt <= c.uwb_timeout_s:
                span = c.theta_max-c.theta_0
                room = (-self.lower-c.theta_0 if direction > 0 else self.upper-c.theta_00)
                output_cap = min(1., max(0., room/span))
                candidate = self.integral+c.ki*normalized*dt
                # 积分保留误差符号，幅值按当前增幅半程的物理余量限幅。
                self.integral = direction*min(max(0., output_cap-abs(p)), abs(candidate))
            self.last_t_ns, self.last_direction = int(t_ns), direction
        pi_output = min(1., max(-1., p+self.integral)) if direction else 0.
        effort = abs(pi_output)  # 严格按abs(e)计算摆幅，左右方向由航向误差决定。
        left = -c.theta_0 - (effort*(c.theta_max-c.theta_0) if direction < 0 else 0.)
        right = c.theta_00 + (effort*(c.theta_max-c.theta_0) if direction > 0 else 0.)
        limited_left = min(self.upper, max(self.lower, left))
        limited_right = min(self.upper, max(self.lower, right))
        # 到点只判断距离；即使鱼头背向目标，也应回中暂停，避免近点继续游走。
        if reached:
            self.integral = 0.
        return NavigationDecision(x, y, yaw, desired, error, distance, effort,
                                  limited_left, limited_right, reached,
                                  limited_left != left or limited_right != right,
                                  pi_output=pi_output)


@dataclass(frozen=True)
class NavigationTailAction(TailAction):
    def __post_init__(self):
        theta, duration = finite_float(self.theta, 'theta'), finite_float(self.t, 't_0')
        if abs(theta) > 30 or duration < 0 or not math.isfinite(duration*1e9):
            raise ValueError('尾鳍须满足abs(theta)<=30，t_0>=0')
        if duration > 0 and round(duration*1e9) < 1:
            raise ValueError('正动作时长不能小于1纳秒')
        object.__setattr__(self, 'theta', theta)
        object.__setattr__(self, 't', duration)


class NavigationTrajectory(RLActionTrajectory):
    def evaluate(self, elapsed):
        # 旧轨迹要求t>0；本脚本对t=0明确规定为下一个PWM周期直接写入目标。
        if self.duration_s == 0:
            return self.target_angles_deg
        return super().evaluate(elapsed)


class NavigationScheduler(RLActionScheduler):
    def schedule(self, agent, action, start_t_ns=None):
        if agent != 'tail' or not isinstance(action, NavigationTailAction):
            raise ValueError('水面导航只接受尾鳍action1')
        if self.active['tail'] is not None:
            raise RuntimeError('尾鳍上一动作尚未完成')
        trajectory = NavigationTrajectory('tail', action, self.previous_theta['tail'], self.centers,
                                          start_override=dict(self.committed_angles))
        validate_trajectory(trajectory, self.limits)
        item = ScheduledRLAction('tail', action, trajectory,
                                 int(self.clock_ns() if start_t_ns is None else start_t_ns))
        self.active['tail'] = item
        return item
