"""分析员：去重、归纳、结构化素材。

独立性设计（支撑 FR-3 并行调度）：
- 分析员是独立 LLM 上下文，仅基于 goal 与 plan 产出结构化要点，
  不读取研究员(Researcher)产出的 evidence，因此可与研究员并行执行。
- 若提供了 StructuredOutput 工具，优先用它约束模型输出为 JSON；
  否则回退到基类 _structured。
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

from .base import BaseAgent


class Analyst(BaseAgent):
    role = "analyst"
    system_prompt = (
        "Role: analyst\n"
        "你是分析员。基于主题与目标，独立产出结构化分析要点（维度、权衡、争议点）。\n"
        '输出 JSON：{"points":["要点1","要点2",...]}'
    )

    def run(self, task: str, state: Dict[str, Any]) -> List[str]:
        goal: str = state["goal"]
        plan = state.get("plan", [])
        plan_hint = ""
        if plan:
            plan_hint = "\n已有子任务规划：" + "; ".join(
                f"{p.get('id')}:{p.get('description')}" for p in plan
            )

        prompt = (
            f"主题：{goal}{plan_hint}\n"
            "请独立分析该主题，去重、归纳，产出结构化要点 "
            'JSON：{"points":["要点1","要点2",...]}'
        )

        so = self._tool("StructuredOutput")
        if so:
            raw = so.execute(prompt=prompt, system=self.system_prompt)
            data = self._parse_json(raw)
        else:
            data = self._structured(prompt)
        return data.get("points", [])
