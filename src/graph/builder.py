"""LangGraph 编排构建：组装节点与边，含质量门条件回流。

主链路（含 FR-3 并行调度）：
    planner → dispatcher
    dispatcher → researcher  ┐
                             ├─ 并行扇出（依赖均已满足，可并发执行）
    dispatcher → analyst     ┘
    researcher  → writer     ┐
                             ├─ writer 扇入（等待两者都完成）
    analyst     → writer     ┘
    writer → critic →(quality_gate)→ writer（回流）/ END（达标或达上限）

节点实现见 nodes.py；本文件只负责「组装」与「上下文打包」。
"""
from __future__ import annotations

import json
from typing import Any, Callable, Dict, List

from langgraph.graph import END, StateGraph

from .nodes import (
    analyst_node,
    critic_node,
    dispatcher_node,
    planner_node,
    quality_gate,
    researcher_node,
    writer_node,
)
from .state import ReportState, SubTask


class WorkflowContext:
    """把模型、各角色 Agent、可观测组件与配置打包，供节点闭包共享。"""

    def __init__(self, model, agents, tracer, cost, config):
        self.model = model
        self.agents = agents          # {"researcher":..., "analyst":..., "writer":..., "critic":...}
        self.tracer = tracer
        self.cost = cost
        self.config = config

    @staticmethod
    def _parse_json(raw: str) -> Dict:
        try:
            return json.loads(raw)
        except Exception:
            import re
            m = re.search(r"\{.*\}", raw, re.S)
            return json.loads(m.group(0)) if m else {}


def build_graph(ctx: WorkflowContext):
    g = StateGraph(ReportState)
    g.add_node("planner", planner_node(ctx))
    g.add_node("dispatcher", dispatcher_node(ctx))
    g.add_node("researcher", researcher_node(ctx))
    g.add_node("analyst", analyst_node(ctx))
    g.add_node("writer", writer_node(ctx))
    g.add_node("critic", critic_node(ctx))

    g.set_entry_point("planner")
    g.add_edge("planner", "dispatcher")

    # FR-3 并行调度：研究员与分析员并行扇出（两者互不依赖）
    g.add_edge("dispatcher", "researcher")
    g.add_edge("dispatcher", "analyst")

    # 写员扇入：等待研究员与分析员都完成（LangGraph 自动等待全部上游）
    g.add_edge("researcher", "writer")
    g.add_edge("analyst", "writer")

    g.add_edge("writer", "critic")
    g.add_conditional_edges("critic", quality_gate(ctx), {"revise": "writer", "done": END})
    return g.compile()
