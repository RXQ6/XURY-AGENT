"""撰写员：基于证据成文，强制引用，禁止无引用断言。"""
from __future__ import annotations

from typing import Any, Dict

from .base import BaseAgent


class Writer(BaseAgent):
    role = "writer"
    system_prompt = (
        "Role: writer\n"
        "你是撰写员。基于黑板上的证据与分析成文，强制引用，禁止无引用断言。\n"
        "输出 Markdown 报告，含标题、章节与 ## References 引用列表。"
    )

    def run(self, task: str, state: Dict[str, Any]) -> str:
        blackboard = state.get("blackboard", {})
        goal: str = state["goal"]
        evidence = blackboard.get("evidence", [])
        analysis = blackboard.get("analysis_points", [])

        refs = "\n".join(
            f"{i+1}. {e.get('title', '')} — {e.get('url', '')}"
            for i, e in enumerate(evidence)
        ) or "（无）"

        analysis_block = "\n".join(f"- {p}" for p in analysis) if analysis else "（无）"

        prompt = (
            f"主题：{goal}\n分析要点：\n{analysis_block}\n"
            f"可用引用：\n{refs}\n"
            "请撰写一份深度研究报告（Markdown），必须基于证据、强制引用，"
            "结尾必须有 '## References' 引用列表。"
        )
        return self._chat(prompt)
