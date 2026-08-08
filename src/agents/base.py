"""Agent 基类。每个角色是独立 LLM 上下文，通过 Blackboard 交换产物，互不污染 prompt。

约定：
- run(task, state) 读取 state['blackboard'] 作为上下文，返回本角色产物；
- 写回 blackboard 由各图节点负责（保持状态更新可追踪、可合并）。
"""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..models.adapter import ModelAdapter


class BaseAgent(ABC):
    role: str = "base"
    system_prompt: str = ""
    tools: List = []

    def __init__(self, model: ModelAdapter, tools: Optional[List] = None, config: Optional[Dict] = None) -> None:
        self.model = model
        self.tools = tools or []
        self.config = config or {}

    # ---- 调用封装 ----
    def _chat(self, prompt: str, system: Optional[str] = None, response_format: Optional[Dict] = None) -> str:
        sys = system if system is not None else self.system_prompt
        return self.model.complete(prompt, system=sys, response_format=response_format)

    def _structured(self, prompt: str, system: Optional[str] = None) -> Dict[str, Any]:
        raw = self._chat(prompt, system=system, response_format={"type": "json_object"})
        return self._parse_json(raw)

    @staticmethod
    def _parse_json(raw: str) -> Dict[str, Any]:
        try:
            return json.loads(raw)
        except Exception:
            m = re.search(r"\{.*\}", raw, re.S)
            if m:
                return json.loads(m.group(0))
            raise

    def _tool(self, name: str):
        for t in self.tools:
            if getattr(t, "name", None) == name:
                return t
        return None

    @abstractmethod
    def run(self, task: str, state: Dict[str, Any]) -> Any:
        ...
