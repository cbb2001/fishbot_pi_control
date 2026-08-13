from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Protocol

from control.runtime.discrete_actions import (
    DiscreteMission,
    FinAction,
    MissionValidationError,
    TailAction,
)


class ActionProvider(Protocol):
    """动作来源协议；未来策略只需返回同一个 DiscreteMission。"""

    def load(self) -> DiscreteMission:
        """读取并返回与存储格式无关的离散任务。"""


class ManualSequenceProvider:
    """从人工 YAML 文件严格读取三路离散动作序列。"""

    _MISSION_KEYS = {"name", "tail_actions", "left_fin_actions", "right_fin_actions"}

    def __init__(self, mission_path: str | Path) -> None:
        self.mission_path = Path(mission_path)

    def load(self) -> DiscreteMission:
        if not self.mission_path.is_file():
            raise MissionValidationError(f"找不到 mission 文件：{self.mission_path}")
        try:
            import yaml
        except ImportError as exc:
            raise MissionValidationError("读取 mission 需要 PyYAML。") from exc

        try:
            with self.mission_path.open("r", encoding="utf-8") as handle:
                raw = yaml.safe_load(handle)
        except Exception as exc:
            raise MissionValidationError(f"读取 mission YAML 失败：{type(exc).__name__}: {exc}") from exc
        if not isinstance(raw, dict):
            raise MissionValidationError("mission YAML 顶层必须是映射。")
        unknown = set(raw) - self._MISSION_KEYS
        missing = self._MISSION_KEYS - set(raw)
        if unknown:
            raise MissionValidationError(f"mission YAML 包含未知字段：{sorted(unknown)}")
        if missing:
            raise MissionValidationError(f"mission YAML 缺少字段：{sorted(missing)}")

        name = raw["name"]
        if not isinstance(name, str) or not name.strip():
            raise MissionValidationError("mission.name 必须是非空字符串。")
        return DiscreteMission(
            name=name.strip(),
            tail_actions=self._parse_tail_actions(raw["tail_actions"]),
            left_fin_actions=self._parse_fin_actions("left_fin_actions", raw["left_fin_actions"]),
            right_fin_actions=self._parse_fin_actions("right_fin_actions", raw["right_fin_actions"]),
        )

    @staticmethod
    def _parse_tail_actions(raw_actions: Any) -> tuple[TailAction, ...]:
        actions = _require_list("tail_actions", raw_actions)
        parsed: list[TailAction] = []
        for index, raw in enumerate(actions):
            item = _require_action_mapping("tail_actions", index, raw, {"t", "A"})
            parsed.append(
                TailAction(
                    t_s=_number(f"tail_actions[{index}].t", item["t"]),
                    amplitude_deg=_number(f"tail_actions[{index}].A", item["A"]),
                )
            )
        return tuple(parsed)

    @staticmethod
    def _parse_fin_actions(name: str, raw_actions: Any) -> tuple[FinAction, ...]:
        actions = _require_list(name, raw_actions)
        parsed: list[FinAction] = []
        for index, raw in enumerate(actions):
            item = _require_action_mapping(name, index, raw, {"t", "b", "r"})
            raw_b = item["b"]
            if isinstance(raw_b, bool) or not isinstance(raw_b, int):
                raise MissionValidationError(f"{name}[{index}].b 必须是整数 1 或 -1。")
            parsed.append(
                FinAction(
                    t_s=_number(f"{name}[{index}].t", item["t"]),
                    direction_b=raw_b,
                    ratio_r=_number(f"{name}[{index}].r", item["r"]),
                )
            )
        return tuple(parsed)


def _require_list(name: str, value: Any) -> list[Any]:
    if not isinstance(value, list):
        raise MissionValidationError(f"{name} 必须是列表，允许使用空列表。")
    return value


def _require_action_mapping(
    sequence_name: str,
    index: int,
    value: Any,
    expected_keys: set[str],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise MissionValidationError(f"{sequence_name}[{index}] 必须是映射。")
    unknown = set(value) - expected_keys
    missing = expected_keys - set(value)
    if unknown:
        raise MissionValidationError(f"{sequence_name}[{index}] 包含未知字段：{sorted(unknown)}")
    if missing:
        raise MissionValidationError(f"{sequence_name}[{index}] 缺少字段：{sorted(missing)}")
    return value


def _number(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MissionValidationError(f"{name} 必须是数值，不能使用布尔值。")
    result = float(value)
    if not math.isfinite(result):
        raise MissionValidationError(f"{name} 必须是有限数。")
    return result
