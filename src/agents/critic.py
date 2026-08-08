"""评审员 Critic：按 rubric 打分、红队、给出修改意见。

裁决：总分 >= 阈值 且 各单项无 0 分 -> 达标；否则生成 feedback 回流 Writer。
"""
from __future__ import annotations

import json
from typing import Any, Dict

from .base import BaseAgent


class Critic(BaseAgent):
    role = "critic"
    system_prompt = (
        "Role: critic\n"
        "你是评审员(Critic)。按 rubric 对报告打分并给出修改意见（红队视角）。\n"
        '输出 JSON：{"score":int(0-40),"breakdown":{"factual_accuracy":0-10,'
        '"citation_completeness":0-10,"structure_clarity":0-10,"coverage":0-10},'
        '"feedback":str,"passed":bool}'
    )

    def run(self, task: str, state: Dict[str, Any]) -> Dict[str, Any]:
        goal: str = state["goal"]
        draft: str = state.get("draft", "")
        threshold = self.config.get("orchestration", {}).get("quality_threshold", 32)

        prompt = (
            f"主题：{goal}\n报告：\n{draft}\n"
            f"请按 rubric（事实准确性/引用完整性/结构清晰度/覆盖度，各 0-10，共 40）打分，"
            f"阈值 {threshold}。输出 JSON："
            '{"score","breakdown":{"factual_accuracy","citation_completeness",'
            '"structure_clarity","coverage"},"feedback","passed"}'
        )
        data = self._structured(prompt)
        score = int(data.get("score", 0))
        breakdown = data.get("breakdown", {})
        # 单项有 0 分视为不达标（防引用幻觉 / 重大缺陷漏判）
        zero_dim = any(int(v) == 0 for v in breakdown.values()) if breakdown else False
        passed = bool(data.get("passed", score >= threshold)) and (score >= threshold) and (not zero_dim)
        data["passed"] = passed
        data["score"] = score
        return data
