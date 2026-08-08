"""统一工具接口（Protocol）。新增工具只需实现 execute + 提供 input_schema。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class Tool(ABC):
    name: str = ""
    description: str = ""
    input_schema: Dict[str, Any] = {"type": "object", "properties": {}}

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """执行工具，返回字符串（通常为 JSON 文本）。"""
        ...
