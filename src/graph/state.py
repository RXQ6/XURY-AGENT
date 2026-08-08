"""贯穿全图的共享状态 ReportState。

- blackboard 用 merge reducer 合并各节点写入（避免并行/回流时覆盖）；
- iteration 用 add reducer 累加（每次 Writer 运行 = 一轮迭代）；
- trace 用 list add reducer 累积每节点耗时/调用（供可观测层导出）。
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, TypedDict


def _merge_dict(left: Dict, right: Dict) -> Dict:
    out = dict(left)
    out.update(right)
    return out


class SubTask(TypedDict):
    id: str
    description: str
    role: str
    depends_on: List[str]


class ReportState(TypedDict):
    goal: str
    plan: List[SubTask]
    blackboard: Annotated[Dict[str, Any], _merge_dict]  # 共享产物：证据/草稿/评审
    draft: str
    critique: Dict[str, Any]                            # {score, breakdown, feedback, passed}
    iteration: Annotated[int, operator.add]
    max_iteration: int
    final_report: str
    trace: Annotated[List[Dict], operator.add]          # 节点耗时/调用记录
