"""通义千问供应商：复用 OpenAI 兼容协议。"""
from __future__ import annotations

import os

from .openai import OpenAIProvider


class QwenProvider(OpenAIProvider):
    provider = "qwen"

    def __init__(
        self,
        model_name: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        **kwargs,
    ) -> None:
        model_name = model_name or os.getenv("QWEN_MODEL", "qwen-plus")
        base_url = base_url or os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        api_key = api_key or os.getenv("QWEN_API_KEY", "")
        super().__init__(model_name=model_name, base_url=base_url, api_key=api_key, **kwargs)
