from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from control.runtime.absolute_discrete_actions import (
    AbsoluteMission, FinAbsoluteAction, MissionValidationError, TailAbsoluteAction,
)


class ActionProvider(Protocol):
    """动作来源接口；未来 RLActionProvider 只需返回相同的领域对象。"""

    def load(self) -> AbsoluteMission: ...


class ManualSequenceProvider:
    """严格读取人工 YAML，但不把 YAML 细节泄漏给调度和执行层。"""

    def __init__(self, mission_path: str | Path) -> None:
        self.mission_path = Path(mission_path)

    def load(self) -> AbsoluteMission:
        try:
            import yaml
            with self.mission_path.open("r", encoding="utf-8") as handle:
                raw = yaml.safe_load(handle)
        except Exception as exc:
            raise MissionValidationError(f"读取 mission YAML 失败：{type(exc).__name__}: {exc}") from exc
        required = {"name", "tail_actions", "left_fin_actions", "right_fin_actions"}
        if not isinstance(raw, dict) or set(raw) != required:
            actual = set(raw) if isinstance(raw, dict) else set()
            raise MissionValidationError(f"mission 顶层字段必须严格为 {sorted(required)}；实际为 {sorted(actual)}。")
        name = raw["name"]
        if not isinstance(name, str) or not name.strip():
            raise MissionValidationError("mission.name 必须是非空字符串。")
        return AbsoluteMission(name.strip(), self._tail(raw["tail_actions"]),
                               self._fins("left_fin_actions", raw["left_fin_actions"]),
                               self._fins("right_fin_actions", raw["right_fin_actions"]))

    @staticmethod
    def _tail(raw: Any) -> tuple[TailAbsoluteAction, ...]:
        keys = {"theta1", "theta2", "theta3", "v1", "v2", "v3"}
        return tuple(TailAbsoluteAction(**{key: _number(f"tail_actions[{i}].{key}", item[key])
                                           for key in keys})
                     for i, item in enumerate(_items("tail_actions", raw, keys)))

    @staticmethod
    def _fins(name: str, raw: Any) -> tuple[FinAbsoluteAction, ...]:
        result = []
        for i, item in enumerate(_items(name, raw, {"theta", "v", "b"})):
            b = item["b"]
            if isinstance(b, bool) or not isinstance(b, int):
                raise MissionValidationError(f"{name}[{i}].b 必须是整数 0 或 1。")
            result.append(FinAbsoluteAction(_number(f"{name}[{i}].theta", item["theta"]),
                                            _number(f"{name}[{i}].v", item["v"]), b))
        return tuple(result)


def _items(name: str, raw: Any, keys: set[str]) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        raise MissionValidationError(f"{name} 必须是列表。")
    for i, item in enumerate(raw):
        if not isinstance(item, dict) or set(item) != keys:
            actual = set(item) if isinstance(item, dict) else set()
            raise MissionValidationError(f"{name}[{i}] 字段必须严格为 {sorted(keys)}；实际为 {sorted(actual)}。")
    return raw


def _number(name: str, value: Any) -> float:
    import math
    if isinstance(value, bool):
        raise MissionValidationError(f"{name} 必须是有限数。")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise MissionValidationError(f"{name} 必须是有限数。") from exc
    if not math.isfinite(result):
        raise MissionValidationError(f"{name} 必须是有限数。")
    return result
