"""LangGraph 编排构建：节点 + 边 + 质量门条件回流。

主链路：planner → dispatcher → researcher → analyst → writer → critic
质量门：critic →(quality_gate)→ writer（回流）/ END（达标或达上限）
"""
from __future__ import annotations

import json
import time
from typing import Any, Callable, Dict, List

from langgraph.graph import END, StateGraph

from ..models.adapter import ModelAdapter
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


# ---------------- 拓扑排序（Dispatcher 用） ----------------
def topo_sort(plan: List[SubTask]) -> List[str]:
    deps: Dict[str, List[str]] = {t["id"]: list(t.get("depends_on", [])) for t in plan}
    indeg = {k: 0 for k in deps}
    for k, vs in deps.items():
        for v in vs:
            if v in indeg:
                indeg[k] += 1
    queue = [k for k, d in indeg.items() if d == 0]
    order: List[str] = []
    while queue:
        n = queue.pop(0)
        order.append(n)
        for k, vs in deps.items():
            if n in vs:
                indeg[k] -= 1
                if indeg[k] == 0:
                    queue.append(k)
    # 若存在环，补回未排到的（降级为串行）
    for k in deps:
        if k not in order:
            order.append(k)
    return order


# ---------------- 节点工厂 ----------------
def _planner(ctx: WorkflowContext) -> Callable:
    def node(state: ReportState):
        t0 = time.time()
        goal = state["goal"]
        prompt = (
            f"目标：{goal}\n请将目标拆解为子任务 DAG，每个子任务指定负责角色"
            "(researcher/analyst/writer)与依赖。输出 JSON："
            '{"plan":[{"id":str,"description":str,"role":str,"depends_on":[]}]}'
        )
        raw = ctx.model.complete(prompt, system="Role: planner\n你是任务分解器(Planner)。",
                                 response_format={"type": "json_object"})
        data = ctx._parse_json(raw)
        plan = data.get("plan", [])
        ms = (time.time() - t0) * 1000
        rec = ctx.tracer.record("planner", ms)
        return {"plan": plan, "trace": [rec]}
    return node


def _dispatcher(ctx: WorkflowContext) -> Callable:
    def node(state: ReportState):
        t0 = time.time()
        order = topo_sort(state.get("plan", []))
        ms = (time.time() - t0) * 1000
        rec = ctx.tracer.record("dispatcher", ms)
        return {"blackboard": {"schedule": order}, "trace": [rec]}
    return node


def _researcher(ctx: WorkflowContext) -> Callable:
    def node(state: ReportState):
        t0 = time.time()
        evidence = ctx.agents["researcher"].run("research", state)
        ms = (time.time() - t0) * 1000
        rec = ctx.tracer.record("researcher", ms, {"evidence_count": len(evidence)})
        return {"blackboard": {"evidence": evidence}, "trace": [rec]}
    return node


def _analyst(ctx: WorkflowContext) -> Callable:
    def node(state: ReportState):
        t0 = time.time()
        points = ctx.agents["analyst"].run("analyze", state)
        analysis_text = "\n".join(f"- {p}" for p in points)
        ms = (time.time() - t0) * 1000
        rec = ctx.tracer.record("analyst", ms, {"point_count": len(points)})
        return {"blackboard": {"analysis_points": points, "analysis": analysis_text}, "trace": [rec]}
    return node


def _writer(ctx: WorkflowContext) -> Callable:
    def node(state: ReportState):
        t0 = time.time()
        draft = ctx.agents["writer"].run("write", state)
        ms = (time.time() - t0) * 1000
        rec = ctx.tracer.record("writer", ms, {"iteration": state.get("iteration", 0) + 1})
        # iteration 用 add reducer：每次 writer 运行 +1
        return {"draft": draft, "iteration": 1, "trace": [rec]}
    return node


def _critic(ctx: WorkflowContext) -> Callable:
    def node(state: ReportState):
        t0 = time.time()
        crit = ctx.agents["critic"].run("critique", state)
        iteration = state.get("iteration", 0)
        max_iter = state.get("max_iteration", 3)
        terminal = crit.get("passed") or iteration >= max_iter
        ms = (time.time() - t0) * 1000
        rec = ctx.tracer.record("critic", ms, {"score": crit.get("score"), "passed": crit.get("passed"), "terminal": terminal})
        upd: Dict[str, Any] = {"critique": crit, "trace": [rec]}
        if terminal:
            upd["final_report"] = state.get("draft", "")
        return upd
    return node


def _quality_gate(ctx: WorkflowContext) -> Callable:
    def gate(state: ReportState) -> str:
        crit = state.get("critique") or {}
        if crit.get("passed"):
            return "done"
        if state.get("iteration", 0) >= state.get("max_iteration", 3):
            return "done"
        return "revise"
    return gate


def build_graph(ctx: WorkflowContext):
    g = StateGraph(ReportState)
    g.add_node("planner", _planner(ctx))
    g.add_node("dispatcher", _dispatcher(ctx))
    g.add_node("researcher", _researcher(ctx))
    g.add_node("analyst", _analyst(ctx))
    g.add_node("writer", _writer(ctx))
    g.add_node("critic", _critic(ctx))

    g.set_entry_point("planner")
    g.add_edge("planner", "dispatcher")
    g.add_edge("dispatcher", "researcher")
    g.add_edge("researcher", "analyst")
    g.add_edge("analyst", "writer")
    g.add_edge("writer", "critic")
    g.add_conditional_edges("critic", _quality_gate(ctx), {"revise": "writer", "done": END})
    return g.compile()
