"""CLI 入口：输入主题 → 输出带引用的深度研究报告（Markdown）。

用法：
    python main.py "对比 LangGraph 与 AutoGen 的优缺点"
    python -m src.main "你的主题" --provider mock --max-iteration 3
    python main.py --goal-file goal.txt --output outputs/my_report.md
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from src.config import get, load_config
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
from src.tools.structured import StructuredOutput


def _slug(s: str) -> str:
    s = re.sub(r"[^\w一-龥]+", "-", s).strip("-")
    return (s[:50] or "topic").lower()


def _seed_corpus(vs: VectorStore, goal: str) -> None:
    """为演示 RAG 检索，向向量库注入少量通用语料（真实场景替换为领域知识库）。"""
    if vs.docs:
        return
    vs.add(f"关于「{goal}」的研究方法：应先界定范围，再交叉验证来源，最后结构化成文。",
           source="builtin-corpus/method")
    vs.add(f"「{goal}」的常见权衡：性能、成本与可扩展性往往相互制约，需结合场景取舍。",
           source="builtin-corpus/tradeoff")
    vs.add(f"撰写「{goal}」报告时，强制引用可溯源证据，可显著降低幻觉与编造风险。",
           source="builtin-corpus/citation")


def main(argv: list[str] | None = None) -> str:
    ap = argparse.ArgumentParser(description="多智能体深度研究报告生成器")
    ap.add_argument("goal", nargs="?", help="报告主题")
    ap.add_argument("--config", default=None, help="配置文件路径（默认 config.yaml）")
    ap.add_argument("--goal-file", default=None, help="从文件读取主题")
    ap.add_argument("--output", default=None, help="输出 Markdown 路径")
    ap.add_argument("--provider", default=None, help="覆盖模型供应商：mock/openai/qwen/hunyuan")
    ap.add_argument("--max-iteration", type=int, default=None, help="覆盖质量门最大迭代次数")
    ap.add_argument("--docs-dir", default=None, help="私有文档目录（.txt/.md/.pdf），用于 RAG 检索增强")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.provider:
        cfg.setdefault("model", {})["provider"] = args.provider
    if args.max_iteration is not None:
        cfg.setdefault("orchestration", {})["max_iteration"] = args.max_iteration

    goal = args.goal
    if not goal and args.goal_file:
        goal = Path(args.goal_file).read_text(encoding="utf-8").strip()
    if not goal:
        ap.error("请提供 goal 参数或 --goal-file")

    out_dir = Path(get(cfg, "output", "dir", default="outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)

    trace_file = get(cfg, "observability", "trace_file")
    cost_file = get(cfg, "observability", "cost_file")
    tracer = Tracer(trace_file)
    cost = CostMeter(cap_cny=get(cfg, "orchestration", "budget_cap_cny", default=2.0), cost_file=cost_file)
    model = build_model(cfg, on_usage=cost)

    vs = VectorStore(persist_path=out_dir / "vector_store.json")
    ingested = vs.add_dir(args.docs_dir) if args.docs_dir else 0
    _seed_corpus(vs, goal)
    web = WebSearch()
    rag = RAGRetriever(vs)
    so = StructuredOutput(model)  # 结构化输出工具：分析师/评审员共用
    agents = {
        "researcher": Researcher(model, tools=[web, rag], config=cfg),
        "analyst": Analyst(model, tools=[so], config=cfg),
        "writer": Writer(model, tools=[], config=cfg),
        "critic": Critic(model, tools=[so], config=cfg),
    }
    ctx = WorkflowContext(model=model, agents=agents, tracer=tracer, cost=cost, config=cfg)
    graph = build_graph(ctx)

    init = {
        "goal": goal,
        "plan": [],
        "blackboard": {},
        "draft": "",
        "critique": {},
        "iteration": 0,
        "max_iteration": get(cfg, "orchestration", "max_iteration", default=3),
        "final_report": "",
        "trace": [],
    }

    print(f"▶ 启动多智能体流水线，主题：{goal}")
    if ingested:
        print(f"   📚 已从私有文档摄入 {ingested} 个片段用于 RAG")
    result = graph.invoke(init)

    report = result.get("final_report") or result.get("draft") or ""
    out_path = Path(args.output) if args.output else (out_dir / f"report_{_slug(goal)}.md")
    out_path.write_text(report, encoding="utf-8")

    crit = result.get("critique", {})
    metrics = {
        "goal": goal,
        "iterations": result.get("iteration", 0),
        "passed": crit.get("passed", False),
        "score": crit.get("score"),
        "cost": cost.report(),
        "trace": tracer.summary(),
    }
    (out_dir / "last_run_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n✅ 报告已生成：{out_path}")
    print(f"   迭代轮数：{metrics['iterations']} ｜ 达标：{metrics['passed']} ｜ 评分：{metrics['score']}")
    c = metrics["cost"]
    print(f"   成本：¥{c['cost_cny']}（token {c['total_tokens']}，调用 {c['calls']} 次）"
          + ("  ⚠️ 超预算！" if c["over_budget"] else ""))
    return report


if __name__ == "__main__":
    main()
