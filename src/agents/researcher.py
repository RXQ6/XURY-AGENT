"""研究员：检索、取证、引用。只输出可溯源的事实与引用。"""
from __future__ import annotations

import json
from typing import Any, Dict, List

from .base import BaseAgent


class Researcher(BaseAgent):
    role = "researcher"
    system_prompt = (
        "Role: researcher\n"
        "你是基于证据的检索研究员。只输出可溯源的事实与引用，禁止编造。\n"
        '调用 WebSearch 与 RAGRetriever 获取资料，整理为 JSON：'
        '{"evidence":[{"title":str,"url":str,"snippet":str}]}'
    )

    def run(self, task: str, state: Dict[str, Any]) -> List[Dict[str, str]]:
        goal: str = state["goal"]
        web = self._tool("WebSearch")
        rag = self._tool("RAGRetriever")

        findings: List[Dict[str, str]] = []
        if web:
            try:
                findings += json.loads(web.execute(query=goal, top_k=3))
            except Exception:
                pass
        if rag:
            try:
                for h in json.loads(rag.execute(query=goal, k=3)):
                    findings.append({
                        "title": h.get("source", goal),
                        "url": h.get("source", ""),
                        "snippet": h.get("text", ""),
                    })
            except Exception:
                pass

        prompt = (
            f"主题：{goal}\n原始检索结果：{json.dumps(findings, ensure_ascii=False)}\n"
            '请产出结构化证据 JSON：{"evidence":[{"title","url","snippet"}]}，必须可溯源。'
        )
        data = self._structured(prompt)
        evidence = data.get("evidence", findings)
        # 过滤掉无 URL 的伪引用（防引用幻觉）
        evidence = [e for e in evidence if isinstance(e, dict) and e.get("url")]
        return evidence
