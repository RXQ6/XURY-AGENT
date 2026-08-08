"""共享黑板 Blackboard：节点间交换证据 / 草稿 / 评审。

在 LangGraph 中与 ReportState.blackboard 对应（使用 merge reducer 合并各节点写入）。
本类提供便捷的读写封装，供 Agent 在 run() 中操作。
"""
from __future__ import annotations

from typing import Any, Dict


class Blackboard:
    def __init__(self, data: Dict[str, Any] | None = None) -> None:
        self._data: Dict[str, Any] = data or {}

    def put(self, key: str, value: Any) -> None:
        self._data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def update(self, d: Dict[str, Any]) -> None:
        self._data.update(d)

    def to_dict(self) -> Dict[str, Any]:
        return self._data

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __repr__(self) -> str:  # pragma: no cover
        return f"Blackboard(keys={list(self._data.keys())})"
