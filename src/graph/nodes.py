"""图节点（Nodes）：planner / dispatcher / researcher / analyst / writer / critic + 质量门。

每个节点工厂接收 WorkflowContext，返回一个 LangGraph 节点函数（接收 ReportState、
返回状态增量）。状态增量通过 state.py 中的 reducer 合并：
- blackboard: _merge_dict（并行/回流时按 key 合并，互不覆盖）
- iteration: operator.add（每次 Writer 运行 = 一轮迭代）
- trace:     operator.add（累积每节点耗时/调用，供可观测层导出）

注意：researcher 与 analyst 互为独立上下文，不互相依赖，因此可由 dispatcher
并行扇出（见 builder.build_graph 的并行边），writer 扇入等待两者都完成。
"""
from __future__ import annotations

import json
import time
from typing import Any, Callable, Dict, List

from .state import ReportState, SubTask


# ---------------- 拓扑排序（Dispatcher 用） ----------------
def topo_sort(plan: List[SubTask]) -> List[str]:
    """对子任务 DAG 做拓扑排序；存在环时降级为串行（补回未排到的节点）。"""
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
def planner_node(ctx) -> Callable:
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


def dispatcher_node(ctx) -> Callable:
    def node(state: ReportState):
        t0 = time.time()
        order = topo_sort(state.get("plan", []))
        ms = (time.time() - t0) * 1000
        rec = ctx.tracer.record("dispatcher", ms)
        return {"blackboard": {"schedule": order}, "trace": [rec]}
    return node


def researcher_node(ctx) -> Callable:
    def node(state: ReportState):
        t0 = time.time()
        evidence = ctx.agents["researcher"].run("research", state)
        ms = (time.time() - t0) * 1000
        rec = ctx.tracer.record("researcher", ms, {"evidence_count": len(evidence)})
        return {"blackboard": {"evidence": evidence}, "trace": [rec]}
    return node


def analyst_node(ctx) -> Callable:
    def node(state: ReportState):
        t0 = time.time()
        # Analyst 基于 goal/plan 独立产出结构化要点，不依赖 researcher 的证据，
        # 因此可与 researcher 并行执行（FR-3）。
        points = ctx.agents["analyst"].run("analyze", state)
        analysis_text = "\n".join(f"- {p}" for p in points)
        ms = (time.time() - t0) * 1000
        rec = ctx.tracer.record("analyst", ms, {"point_count": len(points)})
        return {"blackboard": {"analysis_points": points, "analysis": analysis_text}, "trace": [rec]}
    return node


def writer_node(ctx) -> Callable:
    def node(state: ReportState):
        t0 = time.time()
        draft = ctx.agents["writer"].run("write", state)
        ms = (time.time() - t0) * 1000
        rec = ctx.tracer.record("writer", ms, {"iteration": state.get("iteration", 0) + 1})
        # iteration 用 add reducer：每次 writer 运行 +1
        return {"draft": draft, "iteration": 1, "trace": [rec]}
    return node


def critic_node(ctx) -> Callable:
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


def quality_gate(ctx) -> Callable:
    """质量门条件边：达标或达迭代上限 -> done(END)；否则 -> revise(回流 writer)。"""
    def gate(state: ReportState) -> str:
        crit = state.get("critique") or {}
        if crit.get("passed"):
            return "done"
        if state.get("iteration", 0) >= state.get("max_iteration", 3):
            return "done"
        return "revise"
    return gate
