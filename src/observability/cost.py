"""可观测：token 计量与成本折算。

作为 ModelAdapter 的 on_usage 回调，每次模型调用累加 token 与成本，
并支持预算上限告警（由编排层决定是否中断）。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


class CostMeter:
    def __init__(self, cap_cny: float = 2.0, cost_file: str | Path | None = None) -> None:
        self.in_tokens = 0
        self.out_tokens = 0
        self.cost = 0.0
        self.calls = 0
        self.cap = cap_cny
        self.cost_file = Path(cost_file) if cost_file else None
        self.over_budget = False

    def __call__(self, in_tok: int, out_tok: int, cost: float) -> None:
        self.in_tokens += in_tok
        self.out_tokens += out_tok
        self.cost += cost
        self.calls += 1
        if self.cost > self.cap:
            self.over_budget = True

    def report(self) -> Dict[str, Any]:
        r = {
            "calls": self.calls,
            "in_tokens": self.in_tokens,
            "out_tokens": self.out_tokens,
            "total_tokens": self.in_tokens + self.out_tokens,
            "cost_cny": round(self.cost, 4),
            "budget_cny": self.cap,
            "over_budget": self.over_budget,
        }
        if self.cost_file:
            self.cost_file.parent.mkdir(parents=True, exist_ok=True)
            self.cost_file.write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
        return r
