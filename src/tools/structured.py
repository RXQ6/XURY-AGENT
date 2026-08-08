"""StructuredOutput 工具（报告 §4.3）：约束 LLM 输出为严格 JSON。

用于分析(Analyst)与评审(Critic)的结构化产出。它是对「结构化输出」能力的
统一封装——任何需要模型返回 JSON 的环节都通过该工具，避免各角色重复实现
JSON 解析与容错。返回 JSON 文本，由调用方自行解析。
"""
from __future__ import annotations

from typing import Any, Optional

from .base import Tool
from ..models.adapter import ModelAdapter


class StructuredOutput(Tool):
    name = "StructuredOutput"
    description = "约束 LLM 输出为严格 JSON（用于分析/评审结构化），返回 JSON 文本。"
    input_schema = {
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "description": "要求模型输出的指令"},
            "system": {"type": "string", "description": "可选系统提示"},
        },
        "required": ["prompt"],
    }

    def __init__(self, model: ModelAdapter) -> None:
        self.model = model

    def execute(self, prompt: str, system: Optional[str] = None, **kwargs) -> str:
        """调用模型并要求 JSON 对象输出，返回原始 JSON 文本。"""
        return self.model.complete(prompt, system=system, response_format={"type": "json_object"})
