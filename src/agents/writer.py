"""撰写员：基于证据成文，强制引用，禁止无引用断言。

新增「篇幅深度」(depth) 控制：根据 config.orchestration.depth 决定章节数量与展开深度，
满足"报告要厚、要 15 张以上"的需求（detailed 模式强制 ≥15 个 ## 章节）。
"""
from __future__ import annotations

from typing import Any, Dict

from .base import BaseAgent

# 篇幅档位 -> (最少 ## 章节数, 每章最少段落数, 描述)
_DEPTH_SPEC = {
    "concise": (6, 1, "精简"),
    "standard": (10, 2, "标准"),
    "detailed": (15, 2, "详尽"),
}


class Writer(BaseAgent):
    role = "writer"
    system_prompt = (
        "Role: writer\n"
        "你是撰写员。基于黑板上的证据与分析成文，强制引用，禁止无引用断言。\n"
        "输出 Markdown 报告，含标题、章节与 ## References 引用列表。\n"
        "内容密度要求：每个 ## 二级章节至少 2 段，每段至少 4 个完整句子；"
        "必须包含具体数据、真实案例、可验证引用；禁止空话、套话与重复性描述。"
    )

    def _depth_spec(self, state: Dict[str, Any]):
        cfg = self.config or {}
        depth = (cfg.get("orchestration", {}) or {}).get("depth", "detailed")
        return _DEPTH_SPEC.get(depth, _DEPTH_SPEC["detailed"])

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

        min_sections, min_paras, label = self._depth_spec(state)

        structure_rule = (
            f"【篇幅要求：{label}】报告必须包含至少 {min_sections} 个 `##` 二级章节"
            f"（不含标题与 References），每个章节至少 {min_paras} 段展开论述，每段至少 4 个完整句子。\n"
            "章节应当覆盖（但不限于）：摘要、背景与定义、发展脉络与现状、核心概念拆解、"
            "关键技术/方法、市场规模与数据、典型案例、对比分析、优势与局限、风险与伦理、"
            "趋势预测、政策与监管、应用场景、最佳实践、结论与建议等维度，按需增删以达到章节数。"
        )

        prompt = (
            f"主题：{goal}\n分析要点：\n{analysis_block}\n"
            f"可用引用：\n{refs}\n\n"
            f"{structure_rule}\n\n"
            "请撰写一份深度研究报告（Markdown），严格遵守以下规则：\n"
            "1. 必须用真实证据与数据支撑论点，每个重要论断都要带上引用（如 [1] [2]）。\n"
            "2. 每章要有具体数字、案例或第一手事实，禁止空泛的套话与重复；不得用“本章将介绍”“详见引用”等敷衍表述。\n"
            "3. 每个 `##` 章节至少 2 段，每段至少 4 句；对比分析类章节必须包含 Markdown 表格。\n"
            "4. 全文使用中文，层次用 `#`/`##`/`###` 组织，结尾必须有 `## References` 引用列表，"
            "列出上文「可用引用」中的来源（可补充你已知且合理的来源，但必须真实可查）。\n"
            "5. 不要输出解释性文字，直接输出报告正文。"
        )
        return self._chat(prompt)
