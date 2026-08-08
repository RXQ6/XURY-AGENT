"""DeepSeek 供应商：复用 OpenAI 兼容协议（DeepSeek 提供 OpenAI 兼容接口）。"""
from __future__ import annotations

import os

from .openai import OpenAIProvider


class DeepSeekProvider(OpenAIProvider):
    provider = "deepseek"

    def __init__(
        self,
        model_name: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        **kwargs,
    ) -> None:
        model_name = model_name or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        base_url = base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
        api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
        super().__init__(model_name=model_name, base_url=base_url, api_key=api_key, **kwargs)
