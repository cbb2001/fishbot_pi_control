"""单航段固定 yaw 偏差导航的参数与尾鳍策略。

水面完成 UWB 定位和调向后锁定剩余角差；水下策略仅接收时间，
不接收实时位置或 IMU 航向。传感器、UWB 接收和舵机轨迹沿用原接口。
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, fields

from .rl_sensor_calibration_20260914 import finite_float, wrap180
from .surface_navigation_20261010 import NavigationDecision, NavigationPolicy
from .underwater_navigation_20261026 import UnderwaterConfig as BaseUnderwaterConfig


@dataclass(frozen=True)
class UnderwaterConfig(BaseUnderwaterConfig):
    """保留原配置参数，按 (x, y, h, t) 指定目标及到达超时。"""

    waypoints: tuple[tuple[float, float, float, float | None], ...] | None = None

    def __post_init__(self):
        # 共享标定/时序参数交给基类验证，本入口保留自己的四元组点表和校验。
        base = BaseUnderwaterConfig(**{
            item.name: getattr(self, item.name) for item in fields(BaseUnderwaterConfig)
            if item.name != 'waypoints'
        })
        for item in fields(BaseUnderwaterConfig):
            if item.name != 'waypoints':
                object.__setattr__(self, item.name, getattr(base, item.name))
        if not 0 < self.navigation.delta_yaw < 45:
            raise ValueError('only_one_yaw 的 delta_yaw 必须在 (0,45) 度内，调向完成后进入小角差 PI')
        rows = self.waypoints
        if rows is None:
            rows = ((self.navigation.x_d, self.navigation.y_d, self.h0, None),)
        if not isinstance(rows, (list, tuple)) or not rows:
            raise ValueError('waypoints 必须是至少包含一个 (x,y,h,t) 的列表')
        normalized = []
        for index, row in enumerate(rows, 1):
            if not isinstance(row, (list, tuple)) or len(row) != 4:
                raise ValueError(f'第 {index} 个目标点必须为 (x,y,h,t)，t 为 NULL 或正秒数')
            if any(isinstance(value, bool) for value in row):
                raise ValueError(f'第 {index} 个目标点不能使用布尔值')
            x, y, h = (finite_float(value, f'waypoint_{index}') for value in row[:3])
            if not (0 <= x <= self.navigation.L and 0 <= y <= self.navigation.W):
                raise ValueError(f'第 {index} 个目标点必须在 [0,L] × [0,W] 内')
            if h <= 0:
                raise ValueError(f'第 {index} 个目标点 h 必须大于 0 米，向下为正')
            timeout = row[3]
            if isinstance(timeout, str) and timeout.strip().upper() == 'NULL':
                timeout = None
            if timeout is not None:
                timeout = finite_float(timeout, f'waypoint_{index}_t')
                if (timeout <= 0 or not math.isfinite(timeout*1e9)
                        or round(timeout*1e9) < 1):
                    raise ValueError(f'第 {index} 个目标点 t 必须为 NULL 或可表示纳秒的正秒数')
            normalized.append((x, y, h, timeout))
        if normalized[-1][3] is not None:
            raise ValueError('最后一个目标点（包括只有一个目标点时）的 t 必须为 NULL')
        object.__setattr__(self, 'waypoints', tuple(normalized))

    @classmethod
    def from_robot(cls, robot, **overrides):
        """旧配置为默认值；同名 only_one_yaw 段可覆盖，最后应用命令行。"""
        merged_robot = copy.deepcopy(robot)
        overlay = robot.get('underwater_navigation_20261026_only_one_yaw', {}) or {}
        if not isinstance(overlay, dict):
            raise ValueError('underwater_navigation_20261026_only_one_yaw 必须为映射')
        values = dict(robot.get('underwater_navigation_20261026', {}) or {})
        # 不继承主入口的四元路线；本入口只使用自己的配置段或命令行点表。
        values.pop('waypoints', None)
        values.update(overlay)
        waypoints = values.pop('waypoints', None)
        explicit_waypoints = overrides.pop('waypoints', None)
        if explicit_waypoints is not None:
            waypoints = explicit_waypoints
        merged_robot['underwater_navigation_20261026'] = values
        base = BaseUnderwaterConfig.from_robot(merged_robot, **overrides)
        cfg = cls(**{item.name: getattr(base, item.name) for item in fields(BaseUnderwaterConfig)
                     if item.name != 'waypoints'},
                  waypoints=waypoints)
        validate_safety_depth(cfg, robot['safety']['max_depth_m'])
        return cfg


def validate_safety_depth(cfg, max_depth_m):
    """在启动采集/PWM 前校验所有可能使用的目标深度。"""
    maximum = finite_float(max_depth_m, 'safety.max_depth_m')
    if maximum <= 0:
        raise ValueError('safety.max_depth_m 必须大于 0')
    if cfg.h0 >= maximum:
        raise ValueError('h0 必须低于安全深度')
    for index, (_, _, h, _) in enumerate(cfg.waypoints, 1):
        if h >= maximum:
            raise ValueError(f'第 {index} 个目标点 h 必须低于安全深度 {maximum:g} 米')


class FrozenYawPolicy(NavigationPolicy):
    """沿用尾鳍 PI/限幅；每段仅在 freeze 时计算一次目标 yaw 偏差。

    与原策略相同，ki 按时间继续积分固定偏差；锁定的是误差而非摆幅。
    距离和到点状态也只来自该次水面 UWB，水下估计位置不参与控制。
    """

    def __init__(self, config, calibration):
        super().__init__(config, calibration)
        self.frozen = False

    def freeze(self, x, y, yaw_d_deg, yaw_deg, distance_m):
        error = wrap180(finite_float(yaw_d_deg, 'surface_yaw_d_deg') -
                        finite_float(yaw_deg, 'surface_yaw_deg'))
        if abs(error) >= 45. or abs(error) > self.config.delta_yaw:
            raise ValueError('锁定 yaw 前必须完成水面调向，角差须在 delta_yaw 内且小于 45 度')
        self.x = finite_float(x, 'surface_x')
        self.y = finite_float(y, 'surface_y')
        self.yaw_d_deg = wrap180(finite_float(yaw_d_deg, 'surface_yaw_d_deg'))
        self.yaw_deg = wrap180(finite_float(yaw_deg, 'surface_yaw_deg'))
        self.distance_m = finite_float(distance_m, 'surface_distance_m')
        if self.distance_m < 0:
            raise ValueError('surface_distance_m 不能为负')
        self.yaw_error_deg = error
        self.direction = (self.yaw_error_deg > 0) - (self.yaw_error_deg < 0)
        self.normalized_error = min(1., max(-1., self.yaw_error_deg/self.config.yaw_error_scale_deg))
        self.reached = self.distance_m < self.config.delta_s
        self.integral = 0.
        self.last_t_ns = None
        self.last_direction = 0
        self.frozen = True

    def decide(self, t_ns):
        if not self.frozen:
            raise RuntimeError('须先在水面 freeze yaw 偏差，再执行水下策略')
        c = self.config
        error, direction = self.yaw_error_deg, self.direction
        p = c.kp*self.normalized_error
        if self.last_t_ns is None or t_ns > self.last_t_ns:
            dt = 0. if self.last_t_ns is None else (t_ns-self.last_t_ns)/1e9
            if direction != self.last_direction or direction == 0 or c.ki == 0:
                self.integral = 0.
            elif 0 < dt <= c.uwb_timeout_s:
                span = c.theta_max-c.theta_0
                room = (-self.lower-c.theta_0 if direction > 0 else self.upper-c.theta_00)
                output_cap = min(1., max(0., room/span))
                candidate = self.integral+c.ki*self.normalized_error*dt
                self.integral = direction*min(max(0., output_cap-abs(p)), abs(candidate))
            self.last_t_ns, self.last_direction = int(t_ns), direction
        pi_output = min(1., max(-1., p+self.integral)) if direction else 0.
        effort = abs(pi_output)
        left = -c.theta_0-(effort*(c.theta_max-c.theta_0) if direction < 0 else 0.)
        right = c.theta_00+(effort*(c.theta_max-c.theta_0) if direction > 0 else 0.)
        limited_left = min(self.upper, max(self.lower, left))
        limited_right = min(self.upper, max(self.lower, right))
        if self.reached:
            self.integral = 0.
        return NavigationDecision(self.x, self.y, self.yaw_deg, self.yaw_d_deg, error,
                                  self.distance_m, effort, limited_left, limited_right,
                                  self.reached, limited_left != left or limited_right != right,
                                  pi_output=pi_output)
