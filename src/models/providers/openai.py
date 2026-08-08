"""OpenAI 兼容供应商（OpenAI / 通义千问等共用此实现）。

通义千问 DashScope 提供 OpenAI 兼容接口，因此 qwen provider 直接复用本类，
仅切换 base_url / api_key / model_name（见 qwen.py）。
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List

import requests
from requests.exceptions import (
    ChunkedEncodingError,
    ConnectionError as RequestsConnectionError,
    Timeout,
)

from ..adapter import ModelAdapter

# 真实 LLM API 偶发会断连（RemoteDisconnected / 超时），自动重试可显著提升成功率
_MAX_RETRIES = 3
_RETRY_BACKOFF = 2.0  # 指数退避基数（秒）：2s, 4s, 8s


class OpenAIProvider(ModelAdapter):
    provider = "openai"

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        base_url: str | None = None,
        api_key: str | None = None,
        **kwargs,
    ) -> None:
        model_name = model_name or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        super().__init__(model_name=model_name, **kwargs)
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
        last_err: Exception | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                r = requests.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=90,
                )
                r.raise_for_status()
                data = r.json()
                text = data["choices"][0]["message"]["content"]
                u = data.get("usage", {})
                in_tok = u.get("prompt_tokens") or self.count_tokens(
                    json.dumps(messages, ensure_ascii=False)
                )
                out_tok = u.get("completion_tokens") or self.count_tokens(text)
                return text, in_tok, out_tok
            except (RequestsConnectionError, Timeout, ChunkedEncodingError) as e:
                # 仅网络层异常重试：断连 / 超时 / 分块中断（含 DeepSeek RemoteDisconnected）
                last_err = e
                if attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_BACKOFF * (2 ** (attempt - 1)))
                    continue
                break
            except Exception:
                # 鉴权失败（401）/ 限流（429）等非网络异常，不重试，直接抛出
                raise
        raise last_err if last_err else RuntimeError("LLM 调用失败（未知错误）")
