"""评测脚本：遍历 eval/cases.json，统计通过率 / 成本 / 耗时等指标。

用法：
    python eval/run_eval.py
    python eval/run_eval.py --limit 3 --provider mock

输出：
    eval/report.json        量化评测报表
    eval/last_run_summary.txt  终端摘要
"""
from __future__ import annotations

import sys
from pathlib import Path

# 允许 `python eval/run_eval.py` 直接运行：把项目根加入导入路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import statistics
import time

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

EVAL_DIR = Path(__file__).resolve().parent
ROOT = EVAL_DIR.parent


def run_one(cfg, goal: str, cost: CostMeter, tracer: Tracer):
    model = build_model(cfg, on_usage=cost)
    vs = VectorStore(persist_path=None)  # 评测不持久化向量库
    vs.add(f"关于「{goal}」的研究方法：先界定范围，再交叉验证来源，最后结构化成文。", source="builtin/method")
    web, rag = WebSearch(), RAGRetriever(vs)
    agents = {
        "researcher": Researcher(model, tools=[web, rag], config=cfg),
        "analyst": Analyst(model, tools=[], config=cfg),
        "writer": Writer(model, tools=[], config=cfg),
        "critic": Critic(model, tools=[], config=cfg),
    }
    ctx = WorkflowContext(model=model, agents=agents, tracer=tracer, cost=cost, config=cfg)
    g = build_graph(ctx)
    t0 = time.time()
    res = g.invoke({
        "goal": goal, "plan": [], "blackboard": {}, "draft": "",
        "critique": {}, "iteration": 0,
        "max_iteration": cfg["orchestration"]["max_iteration"],
        "final_report": "", "trace": [],
    })
    elapsed = time.time() - t0
    return res, elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--provider", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--cases", default=str(EVAL_DIR / "cases.json"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.provider:
        cfg.setdefault("model", {})["provider"] = args.provider

    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))["cases"]
    if args.limit:
        cases = cases[: args.limit]

    total = len(cases)
    rows = []
    for c in cases:
        cost = CostMeter(cap_cny=cfg["orchestration"]["budget_cap_cny"])
        tracer = Tracer()
        res, elapsed = run_one(cfg, c["goal"], cost, tracer)
        crit = res.get("critique", {})
        rows.append({
            "id": c["id"],
            "goal": c["goal"],
            "iterations": res.get("iteration", 0),
            "passed": bool(crit.get("passed")),
            "first_pass": bool(crit.get("passed")) and res.get("iteration", 0) == 1,
            "score": crit.get("score"),
            "cost_cny": cost.report()["cost_cny"],
            "tokens": cost.report()["total_tokens"],
            "elapsed_s": round(elapsed, 2),
        })
        print(f"[{c['id']}] {'PASS' if rows[-1]['passed'] else 'FAIL'} "
              f"score={rows[-1]['score']} iter={rows[-1]['iterations']} "
              f"¥{rows[-1]['cost_cny']} {rows[-1]['elapsed_s']}s")

    passed = sum(1 for r in rows if r["passed"])
    first_pass = sum(1 for r in rows if r["first_pass"])
    report = {
        "total": total,
        "final_pass_rate": round(passed / total, 3),
        "first_pass_rate": round(first_pass / total, 3),
        "avg_score": round(statistics.mean(r["score"] for r in rows if r["score"] is not None), 2),
        "avg_cost_cny": round(statistics.mean(r["cost_cny"] for r in rows), 4),
        "avg_tokens": round(statistics.mean(r["tokens"] for r in rows)),
        "avg_iterations": round(statistics.mean(r["iterations"] for r in rows), 2),
        "avg_elapsed_s": round(statistics.mean(r["elapsed_s"] for r in rows), 2),
        "rows": rows,
    }
    (EVAL_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = (
        f"评测完成：{total} 例\n"
        f"  最终达标率   : {report['final_pass_rate']*100:.1f}%\n"
        f"  一次通过率   : {report['first_pass_rate']*100:.1f}%\n"
        f"  平均 Critic 分: {report['avg_score']} / 40\n"
        f"  单篇平均成本 : ¥{report['avg_cost_cny']}\n"
        f"  单篇平均 token: {report['avg_tokens']}\n"
        f"  平均迭代轮数 : {report['avg_iterations']}\n"
        f"  平均耗时     : {report['avg_elapsed_s']}s\n"
    )
    (EVAL_DIR / "last_run_summary.txt").write_text(summary, encoding="utf-8")
    print("\n" + summary)
    print(f"报表已写入：{EVAL_DIR / 'report.json'}")


if __name__ == "__main__":
    main()
