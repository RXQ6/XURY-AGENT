"""轻量向量库（零依赖、可持久化）。

- 使用基于词表的 TF 向量 + 余弦相似度，无需联网安装 chromadb；
- 提供 add() / search()，支撑 RAG 与引用溯源；
- 可持久化到 JSON；生产环境可替换为 chromadb / FAISS（接口保持一致）。
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, List


class VectorStore:
    def __init__(self, persist_path: str | Path | None = None) -> None:
        self.docs: List[Dict] = []  # {id, text, source, vec}
        self.vocab: Dict[str, int] = {}
        self.persist_path = Path(persist_path) if persist_path else None
        if self.persist_path and self.persist_path.exists():
            self.load()

    # ---------- 内部 ----------
    @staticmethod
    def _tokenize(text: str) -> List[str]:
        # 中文按字、英文/数字按词，便于子串与跨语言重叠匹配
        return re.findall(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]", text.lower())

    def _build_vocab(self) -> None:
        self.vocab = {}
        for d in self.docs:
            for t in set(self._tokenize(d["text"])):
                self.vocab.setdefault(t, len(self.vocab))

    def _embed(self, text: str) -> Dict[int, float]:
        vec: Dict[int, float] = {}
        toks = self._tokenize(text)
        n = len(toks) or 1
        for t in toks:
            if t in self.vocab:
                vec[self.vocab[t]] = vec.get(self.vocab[t], 0.0) + 1.0
        for k in vec:
            vec[k] /= n
        return vec

    @staticmethod
    def _cosine(a: Dict[int, float], b: Dict[int, float]) -> float:
        if not a or not b:
            return 0.0
        dot = sum(a[k] * b[k] for k in a if k in b)
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        return dot / (na * nb) if na and nb else 0.0

    # ---------- 对外 API ----------
    def add(self, text: str, source: str = "", doc_id: str | None = None) -> None:
        self.docs.append({
            "id": doc_id or f"d{len(self.docs)}",
            "text": text,
            "source": source,
            "vec": None,
        })
        self._build_vocab()
        for d in self.docs:
            d["vec"] = self._embed(d["text"])
        self.save()

    def search(self, query: str, k: int = 3) -> List[Dict]:
        qvec = self._embed(query)
        scored = []
        for d in self.docs:
            s = self._cosine(qvec, d["vec"])
            if s > 0:
                scored.append({"text": d["text"], "source": d["source"], "score": s})
        scored.sort(key=lambda x: -x["score"])
        return scored[:k]

    def save(self) -> None:
        if not self.persist_path:
            return
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        data = [{"id": d["id"], "text": d["text"], "source": d["source"]} for d in self.docs]
        self.persist_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self) -> None:
        data = json.loads(self.persist_path.read_text(encoding="utf-8"))
        self.docs = [{"id": d["id"], "text": d["text"], "source": d["source"], "vec": None} for d in data]
        self._build_vocab()
        for d in self.docs:
            d["vec"] = self._embed(d["text"])
