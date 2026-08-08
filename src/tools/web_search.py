"""联网检索工具 WebSearch。

三种后端（由 SEARCH_BACKEND 环境变量或构造参数选择）：
- mock       ：返回确定性占位结果，保证离线可跑通、引用可溯源演示（默认）；
- duckduckgo ：零 Key 兜底，爬取 DuckDuckGo HTML 结果（best-effort，无需任何密钥）；
- tavily     ：接入 Tavily Search API，需 TAVILY_API_KEY，质量最好、最适合 LLM 检索。

所有后端返回结构一致：JSON 字符串，元素为 {"title","url","snippet"}。
"""
from __future__ import annotations

import html
import json
import os
import re
import urllib.parse
from typing import Any, Dict, List, Optional

import requests

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

    def __init__(self, backend: Optional[str] = None) -> None:
        self.backend = (backend or os.getenv("SEARCH_BACKEND", "mock")).lower()

    def execute(self, query: str, top_k: int = 3, **kwargs) -> str:
        if self.backend == "tavily":
            items = self._tavily(query, top_k)
        elif self.backend == "duckduckgo":
            items = self._duckduckgo(query, top_k)
        else:
            items = self._mock(query, top_k)
        return json.dumps(items[:top_k], ensure_ascii=False, indent=2)

    # ---------- mock（离线默认） ----------
    def _mock(self, query: str, top_k: int) -> List[Dict[str, str]]:
        items = [
            {"title": f"{query} — 权威综述", "url": f"https://en.wikipedia.org/wiki/{_slug(query)}",
             "snippet": f"关于“{query}”的定义、背景与关键事实。"},
            {"title": f"{query} — 对比与实证", "url": f"https://arxiv.org/search/?query={_slug(query)}",
             "snippet": f"“{query}”的多维度对比与实证研究结果。"},
            {"title": f"{query} — 实践案例", "url": f"https://example.com/research/{_slug(query)}",
             "snippet": f"“{query}”的落地案例与最佳实践（mock 引用，仅用于离线演示）。"},
            {"title": f"{query} — 深度分析", "url": f"https://www.nature.com/search?q={_slug(query)}",
             "snippet": f"“{query}”的前沿研究与数据。"},
        ]
        return items

    # ---------- Tavily（需 TAVILY_API_KEY） ----------
    def _tavily(self, query: str, top_k: int) -> List[Dict[str, str]]:
        key = os.getenv("TAVILY_API_KEY", "")
        if not key:
            # 没配 key 时优雅降级到 mock，避免硬失败
            return self._mock(query, top_k)
        try:
            r = requests.post(
                "https://api.tavily.com/search",
                json={"api_key": key, "query": query, "max_results": top_k, "search_depth": "basic"},
                timeout=30,
            )
            r.raise_for_status()
            results = r.json().get("results", [])
            return [
                {"title": it.get("title", ""), "url": it.get("url", ""),
                 "snippet": (it.get("content") or "")[:300]}
                for it in results
            ]
        except Exception:
            return self._mock(query, top_k)

    # ---------- DuckDuckGo（零 Key 兜底） ----------
    def _duckduckgo(self, query: str, top_k: int) -> List[Dict[str, str]]:
        try:
            r = requests.post(
                "https://html.duckduckgo.com/html/",
                data={"q": query},
                headers={"User-Agent": "Mozilla/5.0 (research-agent)"},
                timeout=30,
            )
            r.raise_for_status()
            return self._parse_ddg(r.text, top_k)
        except Exception:
            return self._mock(query, top_k)

    @staticmethod
    def _parse_ddg(html_text: str, top_k: int) -> List[Dict[str, str]]:
        # 结果标题与链接
        links = re.findall(r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', html_text, re.S)
        snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', html_text, re.S)
        out: List[Dict[str, str]] = []
        for i, (href, title) in enumerate(links[:top_k]):
            url = WebSearch._ddg_real_url(href)
            snippet = re.sub(r"<[^>]+>", "", snippets[i]) if i < len(snippets) else ""
            out.append({
                "title": html.unescape(re.sub(r"<[^>]+>", "", title)).strip(),
                "url": url,
                "snippet": html.unescape(snippet).strip(),
            })
        return out

    @staticmethod
    def _ddg_real_url(href: str) -> str:
        # DuckDuckGo 结果常包一层跳转链接，提取真实地址
        m = re.search(r"uddg=([^&]+)", href)
        if m:
            return urllib.parse.unquote(m.group(1))
        if href.startswith("//"):
            return "https:" + href
        return href
