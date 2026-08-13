from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

from control.safety import SafetyError, ServoLimits, servo_limits_from_config


TAIL_SERVO_IDS = (1, 2, 3)
LEFT_FIN_SERVO_IDS = (4, 5)
RIGHT_FIN_SERVO_IDS = (6, 7)
ALL_SERVO_IDS = TAIL_SERVO_IDS + LEFT_FIN_SERVO_IDS + RIGHT_FIN_SERVO_IDS
SEQUENCE_NAMES = ("tail", "left_fin", "right_fin")


class MissionValidationError(ValueError):
    """表示人工动作或机器人标定不满足执行安全条件。"""


@dataclass(frozen=True)
class TailAction:
    """尾鳍离散动作 action1(t, A)，A 为逻辑向左的有符号角度。"""

    t_s: float
    amplitude_deg: float

    def to_dict(self) -> dict[str, float]:
        return {"t": self.t_s, "A": self.amplitude_deg}


@dataclass(frozen=True)
class FinAction:
    """单侧鳍离散动作 action2/3(t, b, r)。"""

    t_s: float
    direction_b: int
    ratio_r: float

    def to_dict(self) -> dict[str, float | int]:
        return {"t": self.t_s, "b": self.direction_b, "r": self.ratio_r}


@dataclass(frozen=True)
class DiscreteMission:
    """与具体动作来源无关的三路离散动作任务。"""

    name: str
    tail_actions: tuple[TailAction, ...]
    left_fin_actions: tuple[FinAction, ...]
    right_fin_actions: tuple[FinAction, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "tail_actions": [action.to_dict() for action in self.tail_actions],
            "left_fin_actions": [action.to_dict() for action in self.left_fin_actions],
            "right_fin_actions": [action.to_dict() for action in self.right_fin_actions],
        }

    def actions_for(self, sequence_name: str) -> tuple[TailAction | FinAction, ...]:
        if sequence_name == "tail":
            return self.tail_actions
        if sequence_name == "left_fin":
            return self.left_fin_actions
        if sequence_name == "right_fin":
            return self.right_fin_actions
        raise KeyError(f"未知动作序列：{sequence_name}")


@dataclass(frozen=True)
class ServoCalibration:
    """执行离散动作所需的单舵机标定快照。"""

    servo_id: int
    channel: int
    center_deg: float
    min_deg: float
    max_deg: float
    logical_left_sign: float | None = None
    top_reference_deg: float | None = None
    bottom_reference_deg: float | None = None

    @property
    def limits(self) -> ServoLimits:
        return ServoLimits(
            center_angle=self.center_deg,
            min_angle=self.min_deg,
            max_angle=self.max_deg,
        )


@dataclass(frozen=True)
class RobotCalibration:
    """七个舵机按 ID 索引的不可变标定集合。"""

    servos: dict[int, ServoCalibration]

    @property
    def centers_deg(self) -> dict[int, float]:
        return {servo_id: item.center_deg for servo_id, item in self.servos.items()}

    @property
    def channel_to_servo_id(self) -> dict[int, int]:
        return {item.channel: servo_id for servo_id, item in self.servos.items()}


def quintic_smoothstep(u: float) -> float:
    """返回端点位置准确且端点速度为零的五次平滑插值因子。"""

    x = min(1.0, max(0.0, float(u)))
    return 10.0 * x**3 - 15.0 * x**4 + 6.0 * x**5


def smooth_segment(start_deg: float, end_deg: float, progress: float) -> float:
    """在两个角度间执行五次平滑插值。"""

    return float(start_deg) + (float(end_deg) - float(start_deg)) * quintic_smoothstep(progress)


def build_robot_calibration(config: dict[str, Any]) -> RobotCalibration:
    """从 robot.yaml 构造严格标定，不从机械限位猜测物理方向。"""

    raw_channels = config.get("servo", {}).get("channels", [])
    if not isinstance(raw_channels, list):
        raise MissionValidationError("robot.yaml 的 servo.channels 必须是列表。")

    by_id: dict[int, dict[str, Any]] = {}
    channels_seen: set[int] = set()
    for raw in raw_channels:
        if not isinstance(raw, dict) or "servo_id" not in raw or "channel" not in raw:
            continue
        servo_id = int(raw["servo_id"])
        channel = int(raw["channel"])
        if servo_id in by_id:
            raise MissionValidationError(f"robot.yaml 中 servo_id={servo_id} 重复。")
        if channel in channels_seen:
            raise MissionValidationError(f"robot.yaml 中 PCA9685 channel={channel} 重复。")
        by_id[servo_id] = raw
        channels_seen.add(channel)

    missing = [servo_id for servo_id in ALL_SERVO_IDS if servo_id not in by_id]
    if missing:
        raise MissionValidationError(f"robot.yaml 缺少舵机 ID：{missing}。")

    result: dict[int, ServoCalibration] = {}
    for servo_id in ALL_SERVO_IDS:
        raw = by_id[servo_id]
        limits = servo_limits_from_config(config, raw)
        _validate_limits(servo_id, limits)
        direction = raw.get("direction")
        if not isinstance(direction, dict):
            raise MissionValidationError(f"servo_id={servo_id} 缺少 direction 标定。")

        logical_left_sign = None
        top_reference = None
        bottom_reference = None
        if servo_id in TAIL_SERVO_IDS:
            logical_left_sign = _tail_left_sign(servo_id, direction, limits)
        else:
            top_reference = _direction_reference(servo_id, direction, "top_reference_angle")
            bottom_reference = _direction_reference(servo_id, direction, "bottom_reference_angle")
            # top/bottom 是物理方向和完整参考行程标定，可以位于当前软件
            # min_angle/max_angle 安全窗口之外。具体动作按 ratio_r 得到的
            # top_r/bottom_r 仍会在 validate_discrete_mission 中逐项验证，
            # 因而这里不能把物理端点静默截断为软件限位，也不应让较小且
            # 合法的比例动作仅因完整物理参考更宽而无法启动。

        result[servo_id] = ServoCalibration(
            servo_id=servo_id,
            channel=int(raw["channel"]),
            center_deg=limits.center_angle,
            min_deg=limits.min_angle,
            max_deg=limits.max_angle,
            logical_left_sign=logical_left_sign,
            top_reference_deg=top_reference,
            bottom_reference_deg=bottom_reference,
        )
    return RobotCalibration(result)


def validate_discrete_mission(
    mission: DiscreteMission,
    calibration: RobotCalibration,
) -> None:
    """在任何硬件或日志启动前验证完整动作和所有累计终点。"""

    if not isinstance(mission.name, str) or not mission.name.strip():
        raise MissionValidationError("mission.name 必须是非空字符串。")
    if not (mission.tail_actions or mission.left_fin_actions or mission.right_fin_actions):
        raise MissionValidationError("三个动作序列不能同时为空。")

    current_tail = {
        servo_id: calibration.servos[servo_id].center_deg for servo_id in TAIL_SERVO_IDS
    }
    for index, action in enumerate(mission.tail_actions):
        _validate_tail_action(action, index)
        endpoint = evaluate_tail_action(
            action,
            action.t_s,
            current_tail,
            calibration,
        )
        for servo_id, angle in endpoint.items():
            _validate_angle(
                servo_id,
                f"tail_actions[{index}] 累计终点",
                angle,
                calibration.servos[servo_id].limits,
            )
        current_tail = endpoint

    for sequence_name, actions, servo_ids in (
        ("left_fin_actions", mission.left_fin_actions, LEFT_FIN_SERVO_IDS),
        ("right_fin_actions", mission.right_fin_actions, RIGHT_FIN_SERVO_IDS),
    ):
        for index, action in enumerate(actions):
            _validate_fin_action(action, sequence_name, index)
            # 分段均为端点间的单调五次插值，检查中心、top_r、bottom_r 即覆盖全部极值。
            for servo_id in servo_ids:
                item = calibration.servos[servo_id]
                top_r, bottom_r = scaled_fin_references(item, action.ratio_r)
                _validate_angle(servo_id, f"{sequence_name}[{index}] top_r", top_r, item.limits)
                _validate_angle(
                    servo_id,
                    f"{sequence_name}[{index}] bottom_r",
                    bottom_r,
                    item.limits,
                )


def tail_action_boundaries(
    mission: DiscreteMission,
    calibration: RobotCalibration,
) -> tuple[tuple[dict[int, float], dict[int, float]], ...]:
    """返回每个尾鳍动作的累计起点和终点，供调度和测试使用。"""

    current = {servo_id: calibration.servos[servo_id].center_deg for servo_id in TAIL_SERVO_IDS}
    boundaries: list[tuple[dict[int, float], dict[int, float]]] = []
    for action in mission.tail_actions:
        endpoint = evaluate_tail_action(action, action.t_s, current, calibration)
        boundaries.append((dict(current), dict(endpoint)))
        current = endpoint
    return tuple(boundaries)


def evaluate_tail_action(
    action: TailAction,
    elapsed_s: float,
    start_angles_deg: dict[int, float],
    calibration: RobotCalibration,
) -> dict[int, float]:
    """按动作实际起点计算任意时刻的三个尾鳍参考角。"""

    progress = min(1.0, max(0.0, float(elapsed_s) / action.t_s))
    factor = quintic_smoothstep(progress)
    result: dict[int, float] = {}
    for servo_id in TAIL_SERVO_IDS:
        item = calibration.servos[servo_id]
        if item.logical_left_sign is None:
            raise MissionValidationError(f"servo_id={servo_id} 缺少尾鳍逻辑方向。")
        result[servo_id] = (
            float(start_angles_deg[servo_id])
            + item.logical_left_sign * action.amplitude_deg * factor
        )
    return result


def evaluate_fin_action(
    action: FinAction,
    elapsed_s: float,
    servo_ids: tuple[int, int],
    calibration: RobotCalibration,
) -> dict[int, float]:
    """计算单侧根部和末端舵机在指定分段时刻的参考角。"""

    root_id, tip_id = servo_ids
    root = calibration.servos[root_id]
    tip = calibration.servos[tip_id]
    root_top, root_bottom = scaled_fin_references(root, action.ratio_r)
    tip_top, tip_bottom = scaled_fin_references(tip, action.ratio_r)
    # 实机动作定义：b=1 时根部和末端均先向物理下方，b=-1 时均先向物理上方。
    # 每个舵机的角度增减方向仍完全由各自的 top/bottom_reference 标定决定。
    root_first, root_second = (
        (root_bottom, root_top) if action.direction_b == 1 else (root_top, root_bottom)
    )
    tip_first = tip_bottom if action.direction_b == 1 else tip_top
    p = min(1.0, max(0.0, float(elapsed_s) / action.t_s))

    if p <= 0.25:
        root_angle = smooth_segment(root.center_deg, root_first, p / 0.25)
    elif p <= 0.75:
        root_angle = smooth_segment(root_first, root_second, (p - 0.25) / 0.5)
    else:
        root_angle = smooth_segment(root_second, root.center_deg, (p - 0.75) / 0.25)

    if p <= 0.25:
        tip_angle = tip.center_deg
    elif p <= 0.375:
        tip_angle = smooth_segment(tip.center_deg, tip_first, (p - 0.25) / 0.125)
    elif p <= 0.75:
        tip_angle = tip_first
    elif p <= 0.875:
        tip_angle = smooth_segment(tip_first, tip.center_deg, (p - 0.75) / 0.125)
    else:
        tip_angle = tip.center_deg
    return {root_id: root_angle, tip_id: tip_angle}


def scaled_fin_references(item: ServoCalibration, ratio_r: float) -> tuple[float, float]:
    """按各舵机独立标定计算 top_r 和 bottom_r。"""

    if item.top_reference_deg is None or item.bottom_reference_deg is None:
        raise MissionValidationError(f"servo_id={item.servo_id} 缺少物理上下参考角。")
    top_r = item.center_deg + ratio_r * (item.top_reference_deg - item.center_deg)
    bottom_r = item.center_deg + ratio_r * (item.bottom_reference_deg - item.center_deg)
    return top_r, bottom_r


def action_parameters(action: TailAction | FinAction) -> dict[str, float | int]:
    """将动作对象转换为日志使用的原始参数名称。"""

    return action.to_dict()


def action_duration_s(action: TailAction | FinAction) -> float:
    return action.t_s


def sequence_servo_ids(sequence_name: str) -> tuple[int, ...]:
    if sequence_name == "tail":
        return TAIL_SERVO_IDS
    if sequence_name == "left_fin":
        return LEFT_FIN_SERVO_IDS
    if sequence_name == "right_fin":
        return RIGHT_FIN_SERVO_IDS
    raise KeyError(f"未知动作序列：{sequence_name}")


def _validate_tail_action(action: TailAction, index: int) -> None:
    _finite_positive(f"tail_actions[{index}].t", action.t_s)
    _finite(f"tail_actions[{index}].A", action.amplitude_deg)


def _validate_fin_action(action: FinAction, sequence_name: str, index: int) -> None:
    _finite_positive(f"{sequence_name}[{index}].t", action.t_s)
    if isinstance(action.direction_b, bool) or action.direction_b not in (-1, 1):
        raise MissionValidationError(f"{sequence_name}[{index}].b 必须为 1 或 -1。")
    _finite(f"{sequence_name}[{index}].r", action.ratio_r)
    if not 0.0 <= action.ratio_r <= 1.0:
        raise MissionValidationError(f"{sequence_name}[{index}].r 必须在 0..1 内。")


def _validate_limits(servo_id: int, limits: ServoLimits) -> None:
    values: Iterable[float] = (limits.center_angle, limits.min_angle, limits.max_angle)
    if not all(math.isfinite(value) for value in values):
        raise MissionValidationError(f"servo_id={servo_id} 的中心或机械限位不是有限数。")
    if limits.min_angle > limits.max_angle:
        raise MissionValidationError(f"servo_id={servo_id} 的 min_angle 大于 max_angle。")
    _validate_angle(servo_id, "center_angle", limits.center_angle, limits)


def _tail_left_sign(servo_id: int, direction: dict[str, Any], limits: ServoLimits) -> float:
    left = _direction_reference(servo_id, direction, "left_reference_angle")
    right = _direction_reference(servo_id, direction, "right_reference_angle")
    _validate_angle(servo_id, "left_reference_angle", left, limits)
    _validate_angle(servo_id, "right_reference_angle", right, limits)
    if math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12):
        raise MissionValidationError(f"servo_id={servo_id} 的左右参考角不能相同。")
    sign = 1.0 if left > right else -1.0
    expected_increase = "left" if sign > 0 else "right"
    expected_decrease = "right" if sign > 0 else "left"
    if direction.get("angle_increases_toward") != expected_increase:
        raise MissionValidationError(
            f"servo_id={servo_id} 的 angle_increases_toward 与左右参考角矛盾。"
        )
    if direction.get("angle_decreases_toward") != expected_decrease:
        raise MissionValidationError(
            f"servo_id={servo_id} 的 angle_decreases_toward 与左右参考角矛盾。"
        )
    return sign


def _direction_reference(servo_id: int, direction: dict[str, Any], key: str) -> float:
    if key not in direction:
        raise MissionValidationError(f"servo_id={servo_id} 缺少 direction.{key}。")
    try:
        value = float(direction[key])
    except (TypeError, ValueError) as exc:
        raise MissionValidationError(f"servo_id={servo_id} 的 direction.{key} 不是数值。") from exc
    _finite(f"servo_id={servo_id} direction.{key}", value)
    return value


def _validate_angle(servo_id: int, label: str, angle: float, limits: ServoLimits) -> None:
    try:
        limits.validate(float(angle))
    except SafetyError as exc:
        raise MissionValidationError(
            f"servo_id={servo_id} 的 {label}={angle:.6f} 度越界；"
            f"robot.yaml 合法范围为 {limits.min_angle:.6f}..{limits.max_angle:.6f} 度。"
        ) from exc


def _finite(name: str, value: float) -> None:
    if isinstance(value, bool) or not math.isfinite(float(value)):
        raise MissionValidationError(f"{name} 必须是有限数。")


def _finite_positive(name: str, value: float) -> None:
    _finite(name, value)
    if float(value) <= 0.0:
        raise MissionValidationError(f"{name} 必须大于 0。")
