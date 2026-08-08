"""联网检索工具 WebSearch。

- 默认 backend=mock：返回确定性占位结果，保证离线可跑通、引用可溯源演示；
- 真实后端（SerpAPI / Bing 等）可在 backend!="mock" 时扩展，接口保持一致。
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List

from .base import Tool


def _slug(s: str) -> str:
    s = re.sub(r"[^\w一-龥]+", "-", s).strip("-")
    return (s[:40] or "topic").lower()


class WebSearch(Tool):
    name = "WebSearch"
    description = "联网检索，返回标题/摘要/URL 列表"
    input_schema = {
        "type": "object",
        "properties": {"query": {"type": "string"}, "top_k": {"type": "integer"}},
        "required": ["query"],
    }

    def __init__(self, backend: str | None = None) -> None:
        self.backend = backend or os.getenv("SEARCH_BACKEND", "mock")

    def execute(self, query: str, top_k: int = 3, **kwargs) -> str:
        if self.backend == "mock":
            return self._mock(query, top_k)
        # TODO: 接入真实检索后端（SerpAPI / Bing / Google），返回相同结构
        return self._mock(query, top_k)

    def _mock(self, query: str, top_k: int) -> str:
        items: List[Dict[str, str]] = [
            {"title": f"{query} — 权威综述", "url": f"https://en.wikipedia.org/wiki/{_slug(query)}",
             "snippet": f"关于“{query}”的定义、背景与关键事实。"},
            {"title": f"{query} — 对比与实证", "url": f"https://arxiv.org/search/?query={_slug(query)}",
             "snippet": f"“{query}”的多维度对比与实证研究结果。"},
            {"title": f"{query} — 实践案例", "url": f"https://example.com/research/{_slug(query)}",
             "snippet": f"“{query}”的落地案例与最佳实践（mock 引用，仅用于离线演示）。"},
            {"title": f"{query} — 深度分析", "url": f"https://www.nature.com/search?q={_slug(query)}",
             "snippet": f"“{query}”的前沿研究与数据。"},
        ]
        return json.dumps(items[:top_k], ensure_ascii=False, indent=2)
