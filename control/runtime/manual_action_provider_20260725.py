"""把严格的 20260725 人工动作 YAML 转换为与来源无关的领域对象。"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Protocol

from control.runtime.discrete_absolute_actions_20260725 import (
    DiscreteAbsoluteMission,
    FinAction,
    MissionValidationError,
    TailAction,
)


class ActionProvider(Protocol):
    """动作来源抽象；未来 RLActionProvider 只需返回相同领域对象。"""

    def load(self) -> DiscreteAbsoluteMission:
        """加载并返回已经完成结构解析的离散动作任务。"""

        ...


class ManualSequenceProvider:
    """严格读取 20260725 人工动作 YAML。

    本类只负责文件格式到领域对象的转换。动作数学、机械限位、运行时
    previous theta 和调度状态均由其他模块负责，避免执行器依赖 YAML。
    """

    def __init__(self, mission_path: str | Path) -> None:
        """保存待读取的人工任务文件路径，不在构造阶段访问文件。"""

        self.mission_path = Path(mission_path)

    def load(self) -> DiscreteAbsoluteMission:
        """读取 UTF-8 YAML，并拒绝旧速度/周期动作字段及额外字段。"""

        try:
            import yaml

            with self.mission_path.open("r", encoding="utf-8") as handle:
                raw = yaml.safe_load(handle)
        except Exception as exc:
            raise MissionValidationError(
                f"读取 mission YAML 失败：{type(exc).__name__}: {exc}"
            ) from exc

        required = {
            "name",
            "tail_actions",
            "left_fin_actions",
            "right_fin_actions",
        }
        if not isinstance(raw, dict) or set(raw) != required:
            actual = set(raw) if isinstance(raw, dict) else set()
            raise MissionValidationError(
                f"mission 顶层字段必须严格为 {sorted(required)}；"
                f"实际为 {sorted(actual)}。"
            )
        name = raw["name"]
        if not isinstance(name, str) or not name.strip():
            raise MissionValidationError("mission.name 必须是非空字符串。")
        return DiscreteAbsoluteMission(
            name.strip(),
            self._tail_actions(raw["tail_actions"]),
            self._fin_actions("left_fin_actions", raw["left_fin_actions"]),
            self._fin_actions("right_fin_actions", raw["right_fin_actions"]),
        )

    @staticmethod
    def _tail_actions(raw: Any) -> tuple[TailAction, ...]:
        """解析 action1(theta,t)，不接受 theta1/v/frequency 等旧字段。"""

        result: list[TailAction] = []
        for index, item in enumerate(_items("tail_actions", raw, {"theta", "t"})):
            result.append(
                TailAction(
                    _number(f"tail_actions[{index}].theta", item["theta"]),
                    _number(f"tail_actions[{index}].t", item["t"]),
                )
            )
        return tuple(result)

    @staticmethod
    def _fin_actions(name: str, raw: Any) -> tuple[FinAction, ...]:
        """解析 action2/action3(theta,t,b1,b2) 并严格拒绝 bool 伪装整数。"""

        result: list[FinAction] = []
        for index, item in enumerate(
            _items(name, raw, {"theta", "t", "b1", "b2"})
        ):
            b1 = _discrete_integer(f"{name}[{index}].b1", item["b1"], (0, 1))
            b2 = _discrete_integer(f"{name}[{index}].b2", item["b2"], (-1, 1))
            result.append(
                FinAction(
                    _number(f"{name}[{index}].theta", item["theta"]),
                    _number(f"{name}[{index}].t", item["t"]),
                    b1,
                    b2,
                )
            )
        return tuple(result)


def _items(
    name: str,
    raw: Any,
    required_keys: set[str],
) -> list[dict[str, Any]]:
    """验证动作组为列表，且每个动作严格只含指定字段集合。"""

    if not isinstance(raw, list):
        raise MissionValidationError(f"{name} 必须是列表。")
    for index, item in enumerate(raw):
        if not isinstance(item, dict) or set(item) != required_keys:
            actual = set(item) if isinstance(item, dict) else set()
            raise MissionValidationError(
                f"{name}[{index}] 字段必须严格为 {sorted(required_keys)}；"
                f"实际为 {sorted(actual)}。"
            )
    return raw


def _number(name: str, value: Any) -> float:
    """把 YAML 数值转为有限浮点数，并拒绝布尔值伪装成数值。"""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MissionValidationError(f"{name} 必须是有限数。")
    result = float(value)
    if not math.isfinite(result):
        raise MissionValidationError(f"{name} 必须是有限数。")
    return result


def _discrete_integer(name: str, value: Any, allowed: tuple[int, ...]) -> int:
    """读取离散整数标志，并要求其属于调用方给出的有限合法集合。"""

    if isinstance(value, bool) or not isinstance(value, int) or value not in allowed:
        options = "、".join(str(item) for item in allowed)
        raise MissionValidationError(f"{name} 必须是整数 {options} 之一。")
    return int(value)
