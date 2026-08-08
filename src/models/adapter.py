"""统一模型适配层（Model Adapter）。

所有角色/Agent 通过同一个 `chat/complete` 接口调用 LLM，便于：
- 切换供应商（mock / openai / qwen / hunyuan）而不改编排核心；
- 内置 token 计量与成本折算，供可观测层使用。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional


class ModelAdapter(ABC):
    provider: str = "base"

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        price_per_1k_input: float = 0.0,
        price_per_1k_output: float = 0.0,
        on_usage: Optional[Callable[[int, int, float], None]] = None,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.price_in = price_per_1k_input
        self.price_out = price_per_1k_output
        # 每次调用后回调：(input_tokens, output_tokens, cost_cny)
        self.on_usage = on_usage

    @abstractmethod
    def _raw_chat(self, messages: List[Dict[str, str]], response_format=None, **kwargs):
        """子类实现，返回 (text, input_tokens, output_tokens)。"""
        ...

    # ---- 公共能力 ----
    def count_tokens(self, text: str) -> int:
        """粗略 token 估算：中文按字、英文按词，统一 ~4 字符/token。"""
        return max(1, len(text) // 4)

    def chat(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> str:
        text, in_tok, out_tok = self._raw_chat(messages, response_format=response_format, **kwargs)
        cost = in_tok / 1000 * self.price_in + out_tok / 1000 * self.price_out
        if self.on_usage:
            self.on_usage(in_tok, out_tok, cost)
        return text

    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        response_format: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> str:
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, response_format=response_format, **kwargs)
