"""分析员：去重、归纳、结构化素材。读 Blackboard 上的 evidence。"""
from __future__ import annotations

import json
from typing import Any, Dict, List

from .base import BaseAgent


class Analyst(BaseAgent):
    role = "analyst"
    system_prompt = (
        "Role: analyst\n"
        "你是分析员。读取黑板上的证据，去重、归纳、结构化为要点。\n"
        '输出 JSON：{"points":["要点1","要点2",...]}'
    )

    def run(self, task: str, state: Dict[str, Any]) -> List[str]:
        blackboard = state.get("blackboard", {})
        evidence = blackboard.get("evidence", [])
        prompt = (
            f"主题：{state['goal']}\n已有证据：{json.dumps(evidence, ensure_ascii=False)}\n"
            '请去重、归纳，产出结构化要点 JSON：{"points":[...]}'
        )
        data = self._structured(prompt)
        return data.get("points", [])
