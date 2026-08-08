"""RAG 检索工具：包装向量库，返回带来源的文本块。"""
from __future__ import annotations

import json
from typing import Any, Dict

from .base import Tool


class RAGRetriever(Tool):
    name = "RAGRetriever"
    description = "从向量库检索相关段落，返回带来源的文本块"
    input_schema = {
        "type": "object",
        "properties": {"query": {"type": "string"}, "k": {"type": "integer"}},
        "required": ["query"],
    }

    def __init__(self, vector_store) -> None:
        self.vs = vector_store

    def execute(self, query: str, k: int = 3, **kwargs) -> str:
        hits = self.vs.search(query, k)
        if not hits:
            return json.dumps([], ensure_ascii=False)
        out = [{"source": h["source"], "score": round(h["score"], 3), "text": h["text"]} for h in hits]
        return json.dumps(out, ensure_ascii=False, indent=2)
