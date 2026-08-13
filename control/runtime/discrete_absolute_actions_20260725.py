"""定义 20260725 人工定时绝对离散动作的领域对象、标定、轨迹和预验证。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from control.safety import ServoLimits, servo_limits_from_config


TAIL_SERVO_IDS = (1, 2, 3)
LEFT_FIN_SERVO_IDS = (4, 5)
RIGHT_FIN_SERVO_IDS = (6, 7)
ALL_SERVO_IDS = TAIL_SERVO_IDS + LEFT_FIN_SERVO_IDS + RIGHT_FIN_SERVO_IDS
SEQUENCE_NAMES = ("tail", "left_fin", "right_fin")

# 这些默认值只在这一处声明。robot.yaml 可以显式覆盖，但项目默认配置必须保持
# 题目给出的 1～7 号参考姿态、尾鳍范围以及 106/86 耦合标尺。
DEFAULT_ACTION_CONFIG: dict[str, Any] = {
    "tail": {
        "servo_1_center_deg": 85.0,
        "servo_2_center_deg": 95.0,
        "servo_3_center_deg": 95.0,
        "theta_min_deg": -30.0,
        "theta_max_deg": 30.0,
    },
    "left_fin": {
        "root_servo_id": 4,
        "tip_servo_id": 5,
        "root_center_deg": 121.0,
        "tip_center_deg": 94.0,
        "root_reference_span_deg": 106.0,
        "tip_reference_span_deg": 86.0,
    },
    "right_fin": {
        "root_servo_id": 6,
        "tip_servo_id": 7,
        "root_center_deg": 143.0,
        "tip_center_deg": 90.0,
        "root_reference_span_deg": 106.0,
        "tip_reference_span_deg": 86.0,
    },
}


class MissionValidationError(ValueError):
    """人工离散动作、动作参考标定或机械限位不合法。"""


@dataclass(frozen=True)
class TailAction:
    """action1(theta, t)：尾鳍相对固定参考中位的统一目标偏移。"""

    theta: float
    t: float

    def to_dict(self) -> dict[str, float]:
        """返回与人工 mission YAML 一致的可序列化字段。"""

        return {"theta": self.theta, "t": self.t}


@dataclass(frozen=True)
class FinAction:
    """action2/action3(theta, t, b1, b2)：胸鳍根部和尖端配合动作。"""

    theta: float
    t: float
    b1: int
    b2: int

    def to_dict(self) -> dict[str, float | int]:
        """返回与人工 mission YAML 一致的可序列化字段。"""

        return {"theta": self.theta, "t": self.t, "b1": self.b1, "b2": self.b2}


DiscreteAction = TailAction | FinAction


@dataclass(frozen=True)
class DiscreteAbsoluteMission:
    """与 YAML 或未来策略来源无关的三路动作领域对象。"""

    name: str
    tail_actions: tuple[TailAction, ...]
    left_fin_actions: tuple[FinAction, ...]
    right_fin_actions: tuple[FinAction, ...]

    def actions_for(self, sequence_name: str) -> tuple[DiscreteAction, ...]:
        """按内部序列名取得动作；执行层不需要了解 YAML 键名。"""

        mapping: dict[str, tuple[DiscreteAction, ...]] = {
            "tail": self.tail_actions,
            "left_fin": self.left_fin_actions,
            "right_fin": self.right_fin_actions,
        }
        if sequence_name not in mapping:
            raise KeyError(sequence_name)
        return mapping[sequence_name]

    def to_dict(self) -> dict[str, Any]:
        """生成日志和 metadata 使用的完整人工序列快照。"""

        return {
            "name": self.name,
            "tail_actions": [action.to_dict() for action in self.tail_actions],
            "left_fin_actions": [action.to_dict() for action in self.left_fin_actions],
            "right_fin_actions": [action.to_dict() for action in self.right_fin_actions],
        }


@dataclass(frozen=True)
class ServoCalibration:
    """单路舵机通道、配置机械限位以及显式物理端点交集。"""

    servo_id: int
    channel: int
    configured_center_deg: float
    min_deg: float
    max_deg: float
    allowed_min_deg: float
    allowed_max_deg: float
    configured_center_explicit: bool = False

    @property
    def limits(self) -> ServoLimits:
        """返回驱动层仍然使用的 min_angle/max_angle 限位。"""

        return ServoLimits(self.configured_center_deg, self.min_deg, self.max_deg)

    def validate(self, angle_deg: float, label: str) -> float:
        """验证角度，不做 clamp、缩幅或方向修正。"""

        angle = _finite(label, angle_deg)
        if not self.allowed_min_deg <= angle <= self.allowed_max_deg:
            raise MissionValidationError(
                f"servo_id={self.servo_id} 的{label}={angle:g}°超出允许范围"
                f"{self.allowed_min_deg:g}°～{self.allowed_max_deg:g}°。"
            )
        return angle


@dataclass(frozen=True)
class TailReferenceCalibration:
    """action1 使用的三个固定参考中位及 theta 闭区间。"""

    centers_deg: dict[int, float]
    theta_min_deg: float
    theta_max_deg: float


@dataclass(frozen=True)
class FinReferenceCalibration:
    """action2/action3 使用的根部、尖端参考中位和耦合标尺。"""

    root_servo_id: int
    tip_servo_id: int
    root_center_deg: float
    tip_center_deg: float
    root_reference_span_deg: float
    tip_reference_span_deg: float


@dataclass(frozen=True)
class RobotCalibration:
    """20260725 动作数学所需的集中式标定，不包含旧速度参数。"""

    servos: dict[int, ServoCalibration]
    tail: TailReferenceCalibration
    left_fin: FinReferenceCalibration
    right_fin: FinReferenceCalibration

    @property
    def initial_angles_deg(self) -> dict[int, float]:
        """返回任务初始化和安全回中使用的七路参考姿态。"""

        return {
            1: self.tail.centers_deg[1],
            2: self.tail.centers_deg[2],
            3: self.tail.centers_deg[3],
            self.left_fin.root_servo_id: self.left_fin.root_center_deg,
            self.left_fin.tip_servo_id: self.left_fin.tip_center_deg,
            self.right_fin.root_servo_id: self.right_fin.root_center_deg,
            self.right_fin.tip_servo_id: self.right_fin.tip_center_deg,
        }

    @property
    def channel_to_servo_id(self) -> dict[int, int]:
        """返回 PCA9685 channel 到逻辑舵机 ID 的一一映射。"""

        return {servo.channel: servo_id for servo_id, servo in self.servos.items()}

    @property
    def initial_previous_thetas(self) -> dict[str, float]:
        """返回三组运行时 previous theta 的规定初值。"""

        return {
            "tail": 0.0,
            "left_fin": self.left_fin.root_center_deg,
            "right_fin": self.right_fin.root_center_deg,
        }

    def fin_for(self, sequence_name: str) -> FinReferenceCalibration:
        """取得指定胸鳍组标定。"""

        if sequence_name == "left_fin":
            return self.left_fin
        if sequence_name == "right_fin":
            return self.right_fin
        raise KeyError(sequence_name)


@dataclass(frozen=True)
class ActionTrajectory:
    """动作开始时冻结的完整轨迹，历史查询不依赖可变运行状态。"""

    sequence_name: str
    action: DiscreteAction
    previous_theta: float
    current_theta: float
    start_angles_deg: dict[int, float]
    target_angles_deg: dict[int, float]
    duration_s: float
    tip_peak_angle_deg: float | None = None

    @property
    def theta_delta(self) -> float:
        """返回带符号的 current_theta - previous_theta。"""

        return self.current_theta - self.previous_theta

    @property
    def absolute_theta_delta(self) -> float:
        """返回不带方向的根部或尾鳍 theta 变化量。"""

        return abs(self.theta_delta)

    def evaluate(self, elapsed_s: float) -> dict[int, float]:
        """在动作局部时间求七路中的本组参考角。

        起点、终点和保持分支均显式返回精确浮点常量，避免在边界附近
        通过多项式计算得到 84.999999 或 85.000001 一类伪误差。
        """

        elapsed = float(elapsed_s)
        if elapsed <= 0.0:
            return dict(self.start_angles_deg)
        if elapsed >= self.duration_s:
            return dict(self.target_angles_deg)

        if self.sequence_name == "tail":
            if self.previous_theta == self.current_theta:
                return dict(self.target_angles_deg)
            return {
                servo_id: smooth_motion(
                    self.start_angles_deg[servo_id],
                    self.target_angles_deg[servo_id],
                    elapsed,
                    self.duration_s,
                )
                for servo_id in TAIL_SERVO_IDS
            }

        root_id, tip_id = sequence_servo_ids(self.sequence_name)
        if self.previous_theta == self.current_theta:
            # 相邻 theta 相等时 b1/b2 不触发尖端动作，完整保持 t 秒。
            return dict(self.start_angles_deg)

        root_angle = smooth_motion(
            self.start_angles_deg[root_id],
            self.target_angles_deg[root_id],
            elapsed,
            self.duration_s,
        )
        center = self.target_angles_deg[tip_id]
        peak = center if self.tip_peak_angle_deg is None else self.tip_peak_angle_deg
        tip_angle = evaluate_fin_tip_motion(center, peak, elapsed, self.duration_s)
        return {root_id: root_angle, tip_id: tip_angle}


def quintic_smoothstep(u: float) -> float:
    """计算 S(u)=10u³-15u⁴+6u⁵，并在两端返回精确的 0 或 1。"""

    value = float(u)
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    result = 10.0 * value**3 - 15.0 * value**4 + 6.0 * value**5
    # u 极接近 1 时，多项式项相消可能产生约 1e-15 的数值上溢。
    # 这里仅把数学上本应属于 [0,1] 的无量纲系数约束回定义域，不修改
    # 用户动作、机械目标或持续时间，也不属于舵机角度的静默 clamp。
    return min(1.0, max(0.0, result))


def smooth_motion(start: float, target: float, elapsed_s: float, duration_s: float) -> float:
    """用统一五次平滑函数在完整 duration 内从 q0 运动到 q1。"""

    duration = float(duration_s)
    if duration <= 0.0:
        raise ValueError("平滑运动 duration_s 必须大于 0。")
    elapsed = float(elapsed_s)
    if elapsed <= 0.0:
        return float(start)
    if elapsed >= duration:
        return float(target)
    if float(start) == float(target):
        return float(target)
    factor = quintic_smoothstep(elapsed / duration)
    if factor <= 0.0:
        return float(start)
    if factor >= 1.0:
        return float(target)
    result = float(start) + (float(target) - float(start)) * factor
    # 理论轨迹是端点的凸组合；防止最后几个 ulp 因浮点消去越过机械端点。
    return min(max(float(start), float(target)), max(min(float(start), float(target)), result))


def evaluate_fin_tip_motion(
    center_deg: float,
    peak_deg: float,
    elapsed_s: float,
    duration_s: float,
) -> float:
    """计算胸鳍尖端在前半程到峰值、后半程回中位的轨迹。"""

    duration = float(duration_s)
    if duration <= 0.0:
        raise ValueError("胸鳍动作 duration_s 必须大于 0。")
    elapsed = float(elapsed_s)
    if elapsed <= 0.0 or float(center_deg) == float(peak_deg):
        return float(center_deg)
    if elapsed >= duration:
        return float(center_deg)
    half = duration / 2.0
    if elapsed <= half:
        return smooth_motion(center_deg, peak_deg, elapsed, half)
    return smooth_motion(peak_deg, center_deg, elapsed - half, half)


def build_calibration(config: dict[str, Any]) -> RobotCalibration:
    """从 robot.yaml 构造新动作标定并验证全部参考中位。

    现有 servo.channels.center_angle 已与题目给出的 85/95/95/121/94/143/90
    等价，因此默认直接复用。若新配置段显式提供 center 字段，则使用该字段，
    但仍必须通过对应舵机机械限位。
    """

    servos: dict[int, ServoCalibration] = {}
    channels: set[int] = set()
    for raw in config.get("servo", {}).get("channels", []):
        if not isinstance(raw, dict) or "servo_id" not in raw or "channel" not in raw:
            continue
        servo_id = int(raw["servo_id"])
        channel = int(raw["channel"])
        if servo_id in servos or channel in channels:
            raise MissionValidationError("robot.yaml 中 servo_id 或 PCA9685 channel 重复。")
        limits = servo_limits_from_config(config, raw)
        configured_center = _finite(f"servo_id={servo_id}.center_angle", limits.center_angle)
        minimum = _finite(f"servo_id={servo_id}.min_angle", limits.min_angle)
        maximum = _finite(f"servo_id={servo_id}.max_angle", limits.max_angle)
        if minimum > maximum:
            raise MissionValidationError(f"servo_id={servo_id} 的 min_angle 大于 max_angle。")
        allowed_min, allowed_max = _physical_limit_intersection(raw, minimum, maximum, servo_id)
        servo = ServoCalibration(
            servo_id,
            channel,
            configured_center,
            minimum,
            maximum,
            allowed_min,
            allowed_max,
            "center_angle" in raw,
        )
        servo.validate(configured_center, "配置中位")
        servos[servo_id] = servo
        channels.add(channel)

    missing = [servo_id for servo_id in ALL_SERVO_IDS if servo_id not in servos]
    if missing:
        raise MissionValidationError(f"robot.yaml 缺少舵机 ID：{missing}。")

    section = config.get("discrete_absolute_actions_20260725", {})
    if section is None:
        section = {}
    if not isinstance(section, dict):
        raise MissionValidationError("discrete_absolute_actions_20260725 必须是映射。")
    tail_raw = _mapping(section.get("tail", {}), "discrete_absolute_actions_20260725.tail")
    left_raw = _mapping(
        section.get("left_fin", {}),
        "discrete_absolute_actions_20260725.left_fin",
    )
    right_raw = _mapping(
        section.get("right_fin", {}),
        "discrete_absolute_actions_20260725.right_fin",
    )

    tail_centers = {
        servo_id: _reference_center(
            tail_raw,
            f"servo_{servo_id}_center_deg",
            servos[servo_id],
            DEFAULT_ACTION_CONFIG["tail"][f"servo_{servo_id}_center_deg"],
        )
        for servo_id in TAIL_SERVO_IDS
    }
    theta_min = _finite(
        "tail.theta_min_deg",
        tail_raw.get("theta_min_deg", DEFAULT_ACTION_CONFIG["tail"]["theta_min_deg"]),
    )
    theta_max = _finite(
        "tail.theta_max_deg",
        tail_raw.get("theta_max_deg", DEFAULT_ACTION_CONFIG["tail"]["theta_max_deg"]),
    )
    if theta_min > theta_max:
        raise MissionValidationError("tail.theta_min_deg 不能大于 theta_max_deg。")
    if not theta_min <= 0.0 <= theta_max:
        raise MissionValidationError("tail theta 范围必须包含初始化 previous theta=0。")

    tail = TailReferenceCalibration(tail_centers, theta_min, theta_max)
    left = _build_fin_reference("left_fin", left_raw, servos)
    right = _build_fin_reference("right_fin", right_raw, servos)
    if (left.root_servo_id, left.tip_servo_id) != LEFT_FIN_SERVO_IDS:
        raise MissionValidationError("left_fin 必须严格控制 4、5 号舵机。")
    if (right.root_servo_id, right.tip_servo_id) != RIGHT_FIN_SERVO_IDS:
        raise MissionValidationError("right_fin 必须严格控制 6、7 号舵机。")

    calibration = RobotCalibration(servos, tail, left, right)
    for servo_id, angle in calibration.initial_angles_deg.items():
        servos[servo_id].validate(angle, "20260725参考中位")
    return calibration


def build_trajectory(
    sequence_name: str,
    action: DiscreteAction,
    previous_theta: float,
    calibration: RobotCalibration,
) -> ActionTrajectory:
    """用动作开始时尚未提交的 previous theta 构造固定轨迹。"""

    previous = float(previous_theta)
    if sequence_name == "tail":
        if not isinstance(action, TailAction):
            raise TypeError("尾鳍序列必须使用 TailAction。")
        starts = {
            servo_id: calibration.tail.centers_deg[servo_id] + previous
            for servo_id in TAIL_SERVO_IDS
        }
        targets = {
            servo_id: calibration.tail.centers_deg[servo_id] + action.theta
            for servo_id in TAIL_SERVO_IDS
        }
        return ActionTrajectory(
            sequence_name,
            action,
            previous,
            action.theta,
            starts,
            targets,
            action.t,
        )

    if not isinstance(action, FinAction):
        raise TypeError("胸鳍序列必须使用 FinAction。")
    fin = calibration.fin_for(sequence_name)
    starts = {fin.root_servo_id: previous, fin.tip_servo_id: fin.tip_center_deg}
    targets = {fin.root_servo_id: action.theta, fin.tip_servo_id: fin.tip_center_deg}
    delta_abs = abs(action.theta - previous)
    if action.theta == previous or action.b1 == 0:
        peak = fin.tip_center_deg
    else:
        peak = (
            fin.tip_center_deg
            + action.b2
            * delta_abs
            / fin.root_reference_span_deg
            * fin.tip_reference_span_deg
        )
    return ActionTrajectory(
        sequence_name,
        action,
        previous,
        action.theta,
        starts,
        targets,
        action.t,
        peak,
    )


def validate_mission(
    mission: DiscreteAbsoluteMission,
    calibration: RobotCalibration,
) -> None:
    """在任何倒计时、日志、传感器和硬件动作前模拟完整三路任务。"""

    if not isinstance(mission.name, str) or not mission.name.strip():
        raise MissionValidationError("mission.name 必须是非空字符串。")
    if not any(mission.actions_for(name) for name in SEQUENCE_NAMES):
        raise MissionValidationError("三个动作序列不能同时为空。")

    simulated_previous = dict(calibration.initial_previous_thetas)
    for sequence_name in SEQUENCE_NAMES:
        for action_index, action in enumerate(mission.actions_for(sequence_name)):
            _validate_action_parameters(
                sequence_name,
                action_index,
                action,
                calibration,
            )
            previous = simulated_previous[sequence_name]
            trajectory = build_trajectory(sequence_name, action, previous, calibration)
            _validate_trajectory_angles(
                sequence_name,
                action_index,
                trajectory,
                calibration,
            )
            # 这里只更新局部模拟变量，绝不污染运行时调度器的 previous theta。
            simulated_previous[sequence_name] = trajectory.current_theta


def sequence_servo_ids(sequence_name: str) -> tuple[int, ...]:
    """返回动作组严格控制的舵机 ID。"""

    mapping = {
        "tail": TAIL_SERVO_IDS,
        "left_fin": LEFT_FIN_SERVO_IDS,
        "right_fin": RIGHT_FIN_SERVO_IDS,
    }
    if sequence_name not in mapping:
        raise KeyError(sequence_name)
    return mapping[sequence_name]


def reference_config_to_dict(calibration: RobotCalibration) -> dict[str, Any]:
    """生成 metadata 使用的、去除实现对象后的动作参考配置。"""

    return {
        "tail": {
            "servo_1_center_deg": calibration.tail.centers_deg[1],
            "servo_2_center_deg": calibration.tail.centers_deg[2],
            "servo_3_center_deg": calibration.tail.centers_deg[3],
            "theta_min_deg": calibration.tail.theta_min_deg,
            "theta_max_deg": calibration.tail.theta_max_deg,
        },
        "left_fin": {
            "root_servo_id": calibration.left_fin.root_servo_id,
            "tip_servo_id": calibration.left_fin.tip_servo_id,
            "root_center_deg": calibration.left_fin.root_center_deg,
            "tip_center_deg": calibration.left_fin.tip_center_deg,
            "root_reference_span_deg": calibration.left_fin.root_reference_span_deg,
            "tip_reference_span_deg": calibration.left_fin.tip_reference_span_deg,
        },
        "right_fin": {
            "root_servo_id": calibration.right_fin.root_servo_id,
            "tip_servo_id": calibration.right_fin.tip_servo_id,
            "root_center_deg": calibration.right_fin.root_center_deg,
            "tip_center_deg": calibration.right_fin.tip_center_deg,
            "root_reference_span_deg": calibration.right_fin.root_reference_span_deg,
            "tip_reference_span_deg": calibration.right_fin.tip_reference_span_deg,
        },
    }


def _build_fin_reference(
    sequence_name: str,
    raw: dict[str, Any],
    servos: dict[int, ServoCalibration],
) -> FinReferenceCalibration:
    """读取一侧胸鳍参考配置并集中校验。"""

    defaults = DEFAULT_ACTION_CONFIG[sequence_name]
    root_id = _integer(
        f"{sequence_name}.root_servo_id",
        raw.get("root_servo_id", defaults["root_servo_id"]),
    )
    tip_id = _integer(
        f"{sequence_name}.tip_servo_id",
        raw.get("tip_servo_id", defaults["tip_servo_id"]),
    )
    if root_id not in servos or tip_id not in servos:
        raise MissionValidationError(f"{sequence_name} 引用了不存在的舵机 ID。")
    root_center = _reference_center(
        raw,
        "root_center_deg",
        servos[root_id],
        defaults["root_center_deg"],
    )
    tip_center = _reference_center(
        raw,
        "tip_center_deg",
        servos[tip_id],
        defaults["tip_center_deg"],
    )
    root_span = _positive(
        f"{sequence_name}.root_reference_span_deg",
        raw.get("root_reference_span_deg", defaults["root_reference_span_deg"]),
    )
    tip_span = _positive(
        f"{sequence_name}.tip_reference_span_deg",
        raw.get("tip_reference_span_deg", defaults["tip_reference_span_deg"]),
    )
    return FinReferenceCalibration(
        root_id,
        tip_id,
        root_center,
        tip_center,
        root_span,
        tip_span,
    )


def _physical_limit_intersection(
    raw: dict[str, Any],
    configured_min: float,
    configured_max: float,
    servo_id: int,
) -> tuple[float, float]:
    """将显式 top/bottom 物理端点与 min/max 取交集，不假设数值方向。"""

    direction = raw.get("direction", {})
    if not isinstance(direction, dict):
        direction = {}
    has_top = "top_reference_angle" in direction
    has_bottom = "bottom_reference_angle" in direction
    if has_top != has_bottom:
        raise MissionValidationError(
            f"servo_id={servo_id} 必须同时配置 top_reference_angle 和 "
            "bottom_reference_angle。"
        )
    if not has_top:
        return configured_min, configured_max
    top = _finite(f"servo_id={servo_id}.top_reference_angle", direction["top_reference_angle"])
    bottom = _finite(
        f"servo_id={servo_id}.bottom_reference_angle",
        direction["bottom_reference_angle"],
    )
    physical_min, physical_max = min(top, bottom), max(top, bottom)
    allowed_min = max(configured_min, physical_min)
    allowed_max = min(configured_max, physical_max)
    if allowed_min > allowed_max:
        raise MissionValidationError(
            f"servo_id={servo_id} 的物理上下端点与 min_angle/max_angle 没有交集。"
        )
    return allowed_min, allowed_max


def _reference_center(
    raw: dict[str, Any],
    key: str,
    servo: ServoCalibration,
    default_value: float,
) -> float:
    """按“新字段、等价旧字段、规定默认值”的顺序取得参考中位。

    当前 robot.yaml 明确配置了七路 center_angle，因此优先复用而不重复维护
    同义字段。若精简配置没有该旧字段，则回退到本模块唯一常量块中规定的
    85/95/95/121/94/143/90，而不是误用全局通用的 90°安全默认值。
    """

    if key in raw:
        value = raw[key]
    elif servo.configured_center_explicit:
        value = servo.configured_center_deg
    else:
        value = default_value
    return servo.validate(value, key)


def _validate_action_parameters(
    sequence_name: str,
    action_index: int,
    action: DiscreteAction,
    calibration: RobotCalibration,
) -> None:
    """验证动作标量参数，不根据速度推导或修改动作时间。"""

    if sequence_name == "tail":
        if not isinstance(action, TailAction):
            raise MissionValidationError("tail_actions 只能包含 action1(theta,t)。")
        theta = _finite(f"tail_actions[{action_index}].theta", action.theta)
        _positive(f"tail_actions[{action_index}].t", action.t)
        if not calibration.tail.theta_min_deg <= theta <= calibration.tail.theta_max_deg:
            raise MissionValidationError(
                f"tail_actions[{action_index}].theta={theta:g}°超出允许范围"
                f"{calibration.tail.theta_min_deg:g}°～"
                f"{calibration.tail.theta_max_deg:g}°。"
            )
        return

    if not isinstance(action, FinAction):
        raise MissionValidationError(f"{sequence_name}_actions 只能包含胸鳍离散动作。")
    _finite(f"{sequence_name}_actions[{action_index}].theta", action.theta)
    _positive(f"{sequence_name}_actions[{action_index}].t", action.t)
    if (
        isinstance(action.b1, bool)
        or not isinstance(action.b1, int)
        or action.b1 not in (0, 1)
    ):
        raise MissionValidationError(
            f"{sequence_name}_actions[{action_index}].b1 必须为整数 0 或 1。"
        )
    if (
        isinstance(action.b2, bool)
        or not isinstance(action.b2, int)
        or action.b2 not in (-1, 1)
    ):
        raise MissionValidationError(
            f"{sequence_name}_actions[{action_index}].b2 必须为整数 -1 或 1。"
        )


def _validate_trajectory_angles(
    sequence_name: str,
    action_index: int,
    trajectory: ActionTrajectory,
    calibration: RobotCalibration,
) -> None:
    """验证目标和尖端峰值，并给出动作组、索引、前后 theta 与范围。"""

    previous = trajectory.previous_theta
    current = trajectory.current_theta
    for servo_id, angle in trajectory.target_angles_deg.items():
        servo = calibration.servos[servo_id]
        try:
            servo.validate(angle, "计算目标角")
        except MissionValidationError as exc:
            raise MissionValidationError(
                f"动作组={sequence_name}，动作索引={action_index}，"
                f"previous theta={previous:g}°，current theta={current:g}°，"
                f"计算出的{servo_id}号舵机目标角={angle:g}°，"
                f"允许范围={servo.allowed_min_deg:g}°～{servo.allowed_max_deg:g}°。"
            ) from exc

    if sequence_name != "tail" and trajectory.tip_peak_angle_deg is not None:
        tip_id = sequence_servo_ids(sequence_name)[1]
        tip_servo = calibration.servos[tip_id]
        peak = trajectory.tip_peak_angle_deg
        try:
            tip_servo.validate(peak, "尖端峰值角")
        except MissionValidationError as exc:
            raise MissionValidationError(
                f"动作组={sequence_name}，动作索引={action_index}，"
                f"previous theta={previous:g}°，current theta={current:g}°，"
                f"计算出的{tip_id}号舵机目标角={peak:g}°，"
                f"允许范围={tip_servo.allowed_min_deg:g}°～"
                f"{tip_servo.allowed_max_deg:g}°。"
            ) from exc


def _mapping(value: Any, label: str) -> dict[str, Any]:
    """验证配置节点为映射，并保留标签用于生成可定位的中文错误。"""

    if not isinstance(value, dict):
        raise MissionValidationError(f"{label} 必须是映射。")
    return value


def _finite(name: str, value: Any) -> float:
    """把非布尔标量转换为有限浮点数，拒绝 NaN、无穷和非法类型。"""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MissionValidationError(f"{name} 必须是有限数。")
    result = float(value)
    if not math.isfinite(result):
        raise MissionValidationError(f"{name} 必须是有限数。")
    return result


def _positive(name: str, value: Any) -> float:
    """读取严格大于零的有限浮点参数。"""

    result = _finite(name, value)
    if result <= 0.0:
        raise MissionValidationError(f"{name} 必须大于 0。")
    return result


def _integer(name: str, value: Any) -> int:
    """读取真正的整数配置值，并显式拒绝 Python 的布尔整数子类。"""

    if isinstance(value, bool) or not isinstance(value, int):
        raise MissionValidationError(f"{name} 必须是整数。")
    return int(value)
