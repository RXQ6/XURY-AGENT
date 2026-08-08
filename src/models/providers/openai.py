"""OpenAI 兼容供应商（OpenAI / 通义千问等共用此实现）。

通义千问 DashScope 提供 OpenAI 兼容接口，因此 qwen provider 直接复用本类，
仅切换 base_url / api_key / model_name（见 qwen.py）。
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

import requests

from ..adapter import ModelAdapter


class OpenAIProvider(ModelAdapter):
    provider = "openai"

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        base_url: str | None = None,
        api_key: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(model_name, **kwargs)
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")

    def _raw_chat(self, messages: List[Dict[str, str]], response_format=None, **kwargs):
        payload: Dict = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if response_format and response_format.get("type") == "json_object":
            payload["response_format"] = {"type": "json_object"}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        r = requests.post(
            f"{self.base_url}/chat/completions", json=payload, headers=headers, timeout=120
        )
        r.raise_for_status()
        data = r.json()
        text = data["choices"][0]["message"]["content"]
        u = data.get("usage", {})
        in_tok = u.get("prompt_tokens") or self.count_tokens(json.dumps(messages, ensure_ascii=False))
        out_tok = u.get("completion_tokens") or self.count_tokens(text)
        return text, in_tok, out_tok
