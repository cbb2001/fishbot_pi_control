from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from control.safety import SafetyError, ServoLimits, servo_limits_from_config


TAIL_SERVO_IDS = (1, 2, 3)
LEFT_FIN_SERVO_IDS = (4, 5)
RIGHT_FIN_SERVO_IDS = (6, 7)
ALL_SERVO_IDS = TAIL_SERVO_IDS + LEFT_FIN_SERVO_IDS + RIGHT_FIN_SERVO_IDS
SEQUENCE_NAMES = ("tail", "left_fin", "right_fin")


class MissionValidationError(ValueError):
    """人工绝对角度任务或机器人配置不满足执行条件。"""


@dataclass(frozen=True)
class TailAbsoluteAction:
    """action1：三个尾鳍舵机分别运动到给定的绝对 PCA9685 角度。"""

    theta1: float
    theta2: float
    theta3: float
    v1: float
    v2: float
    v3: float

    def targets(self) -> dict[int, float]:
        return {1: self.theta1, 2: self.theta2, 3: self.theta3}

    def speeds(self) -> dict[int, float]:
        return {1: self.v1, 2: self.v2, 3: self.v3}

    def to_dict(self) -> dict[str, float]:
        return {"theta1": self.theta1, "theta2": self.theta2, "theta3": self.theta3,
                "v1": self.v1, "v2": self.v2, "v3": self.v3}


@dataclass(frozen=True)
class FinAbsoluteAction:
    """action2/3：根部运动到绝对角度，尖端由 b 决定是否配合。"""

    theta: float
    v: float
    b: int

    def to_dict(self) -> dict[str, float | int]:
        return {"theta": self.theta, "v": self.v, "b": self.b}


AbsoluteAction = TailAbsoluteAction | FinAbsoluteAction


@dataclass(frozen=True)
class AbsoluteMission:
    """与 YAML 或未来 RL 来源无关的三路绝对动作任务。"""

    name: str
    tail_actions: tuple[TailAbsoluteAction, ...]
    left_fin_actions: tuple[FinAbsoluteAction, ...]
    right_fin_actions: tuple[FinAbsoluteAction, ...]

    def actions_for(self, sequence_name: str) -> tuple[AbsoluteAction, ...]:
        mapping = {"tail": self.tail_actions, "left_fin": self.left_fin_actions,
                   "right_fin": self.right_fin_actions}
        if sequence_name not in mapping:
            raise KeyError(sequence_name)
        return mapping[sequence_name]

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name,
                "tail_actions": [item.to_dict() for item in self.tail_actions],
                "left_fin_actions": [item.to_dict() for item in self.left_fin_actions],
                "right_fin_actions": [item.to_dict() for item in self.right_fin_actions]}


@dataclass(frozen=True)
class ServoCalibration:
    servo_id: int
    channel: int
    center_deg: float
    min_deg: float
    max_deg: float

    @property
    def limits(self) -> ServoLimits:
        return ServoLimits(self.center_deg, self.min_deg, self.max_deg)


@dataclass(frozen=True)
class CouplingCalibration:
    root_reference_span_deg: float
    tip_reference_span_deg: float


@dataclass(frozen=True)
class AbsoluteRobotCalibration:
    servos: dict[int, ServoCalibration]
    tail_max_speed_deg_s: float
    fin_max_speed_deg_s: float
    left_coupling: CouplingCalibration
    right_coupling: CouplingCalibration

    @property
    def centers_deg(self) -> dict[int, float]:
        return {key: value.center_deg for key, value in self.servos.items()}

    @property
    def channel_to_servo_id(self) -> dict[int, int]:
        return {item.channel: key for key, item in self.servos.items()}


@dataclass(frozen=True)
class ActionTrajectory:
    """动作实际起点确定后得到的完整数学轨迹。"""

    sequence_name: str
    action: AbsoluteAction
    start_angles_deg: dict[int, float]
    target_angles_deg: dict[int, float]
    servo_durations_s: dict[int, float]
    duration_s: float
    peak_angle_deg: float | None = None

    def evaluate(self, elapsed_s: float) -> dict[int, float]:
        if self.sequence_name == "tail":
            return {servo_id: smooth_motion(self.start_angles_deg[servo_id], target,
                                             elapsed_s, self.servo_durations_s[servo_id])
                    for servo_id, target in self.target_angles_deg.items()}
        root_id, tip_id = sequence_servo_ids(self.sequence_name)
        root = smooth_motion(self.start_angles_deg[root_id], self.target_angles_deg[root_id],
                             elapsed_s, self.duration_s)
        center = self.target_angles_deg[tip_id]
        peak = center if self.peak_angle_deg is None else self.peak_angle_deg
        tip = evaluate_tip_motion(center, peak, elapsed_s, self.duration_s)
        return {root_id: root, tip_id: tip}


def quintic_smoothstep(u: float) -> float:
    """五次平滑函数；位置、速度和加速度在两端均连续归零。"""

    x = min(1.0, max(0.0, float(u)))
    return 10.0 * x**3 - 15.0 * x**4 + 6.0 * x**5


def smooth_motion(start: float, target: float, elapsed_s: float, duration_s: float) -> float:
    """按指定持续时间平滑移动；零持续时间直接保持绝对目标。"""

    if duration_s <= 0.0:
        return float(target)
    return float(start) + (float(target) - float(start)) * quintic_smoothstep(elapsed_s / duration_s)


def evaluate_tip_motion(center: float, peak: float, elapsed_s: float, duration_s: float) -> float:
    """计算 5/7 号舵机进入峰值、保持和回中三段轨迹。"""

    if duration_s <= 0.0 or peak == center:
        return float(center)
    p = min(1.0, max(0.0, elapsed_s / duration_s))
    if p <= 0.25:
        return smooth_motion(center, peak, p, 0.25)
    if p < 0.75:
        return float(peak)
    return smooth_motion(peak, center, p - 0.75, 0.25)


def build_absolute_calibration(config: dict[str, Any]) -> AbsoluteRobotCalibration:
    """只从 robot.yaml 读取七路中位、机械限位、速度和耦合参数。"""

    servos: dict[int, ServoCalibration] = {}
    channels: set[int] = set()
    for raw in config.get("servo", {}).get("channels", []):
        if not isinstance(raw, dict) or "servo_id" not in raw or "channel" not in raw:
            continue
        servo_id, channel = int(raw["servo_id"]), int(raw["channel"])
        if servo_id in servos or channel in channels:
            raise MissionValidationError("robot.yaml 中 servo_id 或 PCA9685 channel 重复。")
        limits = servo_limits_from_config(config, raw)
        for value in (limits.center_angle, limits.min_angle, limits.max_angle):
            _finite("舵机标定角度", value)
        if limits.min_angle > limits.max_angle:
            raise MissionValidationError(f"servo_id={servo_id} 的机械限位顺序错误。")
        _validate_angle(servo_id, limits.center_angle, limits, "center_angle")
        servos[servo_id] = ServoCalibration(servo_id, channel, limits.center_angle,
                                            limits.min_angle, limits.max_angle)
        channels.add(channel)
    missing = [item for item in ALL_SERVO_IDS if item not in servos]
    if missing:
        raise MissionValidationError(f"robot.yaml 缺少舵机 ID：{missing}。")
    cfg = config.get("servo", {}).get("discrete_absolute_actions", {})
    tail_max = _positive("tail_max_speed_deg_s", cfg.get("tail_max_speed_deg_s", 150.0))
    fin_max = _positive("fin_max_speed_deg_s", cfg.get("fin_max_speed_deg_s", 138.0))
    coupling = config.get("discrete_action_coupling", {})
    left = _coupling("left", coupling.get("left", {}))
    right = _coupling("right", coupling.get("right", {}))
    return AbsoluteRobotCalibration(servos, tail_max, fin_max, left, right)


def build_trajectory(sequence_name: str, action: AbsoluteAction,
                     start_angles_deg: dict[int, float],
                     calibration: AbsoluteRobotCalibration) -> ActionTrajectory:
    """从动作实际起点生成轨迹；v 是名义平均速度，时间严格为 abs(delta)/v。"""

    if sequence_name == "tail":
        if not isinstance(action, TailAbsoluteAction):
            raise TypeError("尾鳍序列必须使用 TailAbsoluteAction。")
        targets, speeds = action.targets(), action.speeds()
        durations = {servo_id: abs(targets[servo_id] - start_angles_deg[servo_id]) /
                     speeds[servo_id] for servo_id in TAIL_SERVO_IDS}
        # 三路速度不被同步修改；总时间仅取独立运动时间的最大值。
        return ActionTrajectory(sequence_name, action, dict(start_angles_deg), targets,
                                durations, max(durations.values()))
    if not isinstance(action, FinAbsoluteAction):
        raise TypeError("侧鳍序列必须使用 FinAbsoluteAction。")
    root_id, tip_id = sequence_servo_ids(sequence_name)
    delta = action.theta - start_angles_deg[root_id]
    duration = abs(delta) / action.v
    center = calibration.servos[tip_id].center_deg
    mapping = calibration.left_coupling if sequence_name == "left_fin" else calibration.right_coupling
    # 正负方向统一包含在 delta 内，禁止因机械镜像另行反号。
    peak = center if action.b == 0 else center + delta / mapping.root_reference_span_deg * mapping.tip_reference_span_deg #/ 2
    return ActionTrajectory(sequence_name, action, dict(start_angles_deg),
                            {root_id: action.theta, tip_id: center},
                            {root_id: duration, tip_id: duration}, duration, peak)


def validate_absolute_mission(mission: AbsoluteMission,
                              calibration: AbsoluteRobotCalibration) -> None:
    """在硬件、倒计时、正式日志和传感器启动前模拟并验证完整任务。"""

    if not mission.name.strip() or not any(mission.actions_for(x) for x in SEQUENCE_NAMES):
        raise MissionValidationError("mission 名称不能为空，且三个序列不能同时为空。")
    states = {"tail": {i: calibration.servos[i].center_deg for i in TAIL_SERVO_IDS},
              "left_fin": {i: calibration.servos[i].center_deg for i in LEFT_FIN_SERVO_IDS},
              "right_fin": {i: calibration.servos[i].center_deg for i in RIGHT_FIN_SERVO_IDS}}
    for sequence_name in SEQUENCE_NAMES:
        for index, action in enumerate(mission.actions_for(sequence_name)):
            _validate_action(sequence_name, index, action, calibration)
            trajectory = build_trajectory(sequence_name, action, states[sequence_name], calibration)
            for servo_id, angle in trajectory.target_angles_deg.items():
                _validate_angle(servo_id, angle, calibration.servos[servo_id].limits,
                                f"{sequence_name} 第{index + 1}个动作目标")
            if trajectory.peak_angle_deg is not None:
                tip_id = sequence_servo_ids(sequence_name)[1]
                side = "左侧鳍" if sequence_name == "left_fin" else "右侧鳍"
                try:
                    calibration.servos[tip_id].limits.validate(trajectory.peak_angle_deg)
                except SafetyError as exc:
                    limits = calibration.servos[tip_id].limits
                    raise MissionValidationError(
                        f"{side}第{index + 1}个动作计算得到{tip_id}号舵机目标角"
                        f"{trajectory.peak_angle_deg:.1f}°，超出config/robot.yaml规定范围"
                        f"{limits.min_angle:g}°～{limits.max_angle:g}°。") from exc
            states[sequence_name] = dict(trajectory.target_angles_deg)


def sequence_servo_ids(name: str) -> tuple[int, ...]:
    return {"tail": TAIL_SERVO_IDS, "left_fin": LEFT_FIN_SERVO_IDS,
            "right_fin": RIGHT_FIN_SERVO_IDS}[name]


def _validate_action(name: str, index: int, action: AbsoluteAction,
                     calibration: AbsoluteRobotCalibration) -> None:
    if name == "tail":
        assert isinstance(action, TailAbsoluteAction)
        for servo_id, speed in action.speeds().items():
            _speed(f"tail_actions[{index}].v{servo_id}", speed, calibration.tail_max_speed_deg_s)
        return
    assert isinstance(action, FinAbsoluteAction)
    _speed(f"{name}_actions[{index}].v", action.v, calibration.fin_max_speed_deg_s)
    if isinstance(action.b, bool) or action.b not in (0, 1):
        raise MissionValidationError(f"{name}_actions[{index}].b 必须为整数 0 或 1。")


def _coupling(side: str, raw: Any) -> CouplingCalibration:
    if not isinstance(raw, dict):
        raise MissionValidationError(f"discrete_action_coupling.{side} 必须是映射。")
    return CouplingCalibration(
        _positive(f"{side}.root_reference_span_deg", raw.get("root_reference_span_deg", 106.0)),
        _positive(f"{side}.tip_reference_span_deg", raw.get("tip_reference_span_deg", 90.0)))


def _speed(name: str, value: float, maximum: float) -> None:
    speed = _positive(name, value)
    if speed > maximum:
        raise MissionValidationError(f"{name}={speed:g} deg/s 超过 robot.yaml 上限 {maximum:g} deg/s。")


def _validate_angle(servo_id: int, angle: float, limits: ServoLimits, label: str) -> None:
    _finite(label, angle)
    try:
        limits.validate(float(angle))
    except SafetyError as exc:
        raise MissionValidationError(
            f"servo_id={servo_id} 的{label}={angle:g}°超出config/robot.yaml规定范围"
            f"{limits.min_angle:g}°～{limits.max_angle:g}°。") from exc


def _finite(name: str, value: Any) -> float:
    if isinstance(value, bool):
        raise MissionValidationError(f"{name} 必须是有限数。")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise MissionValidationError(f"{name} 必须是有限数。") from exc
    if not math.isfinite(result):
        raise MissionValidationError(f"{name} 必须是有限数。")
    return result


def _positive(name: str, value: Any) -> float:
    result = _finite(name, value)
    if result <= 0.0:
        raise MissionValidationError(f"{name} 必须大于 0。")
    return result
