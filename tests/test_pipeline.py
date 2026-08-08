"""流水线测试：端到端闭环、质量门迭代、向量库、引用防泄漏。

运行：pytest -q
（默认使用 mock provider，无需任何 API Key）
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.graph.builder import WorkflowContext, build_graph
from src.agents.analyst import Analyst
from src.agents.critic import Critic
from src.agents.researcher import Researcher
from src.agents.writer import Writer
from src.memory.vector_store import VectorStore
from src.models.factory import build_model
from src.observability.cost import CostMeter
from src.observability.tracer import Tracer
from src.tools.rag import RAGRetriever
from src.tools.web_search import WebSearch


def _build(threshold: int | None = None, max_iteration: int = 3):
    cfg = load_config()
    if threshold is not None:
        cfg["orchestration"]["quality_threshold"] = threshold
    cfg["orchestration"]["max_iteration"] = max_iteration
    model = build_model(cfg)
    vs = VectorStore(persist_path=None)
    vs.add("关于 Agent 的记忆机制：分短期上下文与长期向量库两类。", source="builtin/memory")
    web, rag = WebSearch(), RAGRetriever(vs)
    agents = {
        "researcher": Researcher(model, tools=[web, rag], config=cfg),
        "analyst": Analyst(model, tools=[], config=cfg),
        "writer": Writer(model, tools=[], config=cfg),
        "critic": Critic(model, tools=[], config=cfg),
    }
    ctx = WorkflowContext(model=model, agents=agents, tracer=Tracer(), cost=CostMeter(), config=cfg)
    return build_graph(ctx)


def test_end_to_end_mock():
    g = _build()
    res = g.invoke({
        "goal": "对比 LangGraph 与 AutoGen 的优缺点", "plan": [], "blackboard": {},
        "draft": "", "critique": {}, "iteration": 0, "max_iteration": 3,
        "final_report": "", "trace": [],
    })
    assert res["final_report"], "应产出最终报告"
    assert "## References" in res["final_report"], "报告必须含引用列表"
    assert res["critique"].get("passed") is True, "mock 默认一次通过"
    assert res["iteration"] == 1, "mock 应一轮达标"


def test_quality_gate_iterates_to_max():
    # 阈值 41 > mock 满分 40，必然不达标 -> 回流直到 max_iteration
    g = _build(threshold=41, max_iteration=3)
    res = g.invoke({
        "goal": "测试迭代回路", "plan": [], "blackboard": {}, "draft": "",
        "critique": {}, "iteration": 0, "max_iteration": 3, "final_report": "", "trace": [],
    })
    assert res["iteration"] == 3, "应迭代到上限 3 次（防死循环）"
    assert res["final_report"], "达上限后应产出最优草稿"


def test_report_has_no_prompt_leakage():
    g = _build()
    res = g.invoke({
        "goal": "RAG 系统如何降低幻觉", "plan": [], "blackboard": {}, "draft": "",
        "critique": {}, "iteration": 0, "max_iteration": 3, "final_report": "", "trace": [],
    })
    report = res["final_report"]
    # 回归：mock 修好前会把原始检索 JSON 灌进报告
    assert "原始检索结果" not in report
    assert "请产出结构化证据" not in report


def test_vector_store_search():
    vs = VectorStore(persist_path=None)
    vs.add("LangGraph 用状态机编排多智能体。", source="doc/langgraph")
    vs.add("AutoGen 强调对话式多智能体协作。", source="doc/autogen")
    vs.add("今天天气不错。", source="doc/weather")
    hits = vs.search("多智能体编排", k=2)
    assert hits, "应检索到结果"
    sources = [h["source"] for h in hits]
    assert "doc/langgraph" in sources
    assert "doc/weather" not in sources  # 无关文档不应进入 top-k
