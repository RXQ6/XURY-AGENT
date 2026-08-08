"""FastAPI 接入层：把多智能体流水线包装成网页应用后端。

对应设计报告 §3.1「接入层 Interface = CLI / API」——核心编排逻辑全部复用 src/，
本文件只做 HTTP 包装与实时进度推送。

路由：
- GET  /                      静态前端页面（app/static/index.html）
- POST /api/generate/stream   SSE 流式：逐个推送节点执行事件，末帧推送 done(报告+指标)
- POST /api/generate          JSON：一次性返回报告与指标
（SSE 改用 POST 以在请求体中携带可选 api_key / base_url，避免密钥出现在 URL 与日志）

启动：uvicorn app.server:app --host 127.0.0.1 --port 8000

实现要点：流水线在独立 threading.Thread 中同步运行（LangGraph 同步 invoke 需自己的事件循环，
不能在 asyncio.to_thread 的协程线程里跑，否则会与 uvicorn 主循环冲突）。节点事件通过
loop.call_soon_threadsafe 安全推给 SSE 的 asyncio.Queue。
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, Response
from pydantic import BaseModel

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
from src.tools.structured import StructuredOutput
from src.tools.web_search import WebSearch
from src.tools.export import export_report

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
STATIC = BASE / "static"

app = FastAPI(title="多智能体深度研究报告生成器", version="1.0")


class GenerateBody(BaseModel):
    """生成请求体。api_key / base_url 可选：网页端临时填入，仅 openai / qwen 生效，
    且不写进 URL（避免密钥出现在地址栏与服务器日志）。"""
    goal: str
    provider: str = "mock"
    max_iteration: int = 3
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    docs_dir: Optional[str] = None


class StreamingTracer(Tracer):
    """在原有 Tracer 基础上，把每次 record 事件经 asyncio.Queue 推给 SSE 消费者。"""

    def __init__(self, queue: Optional[asyncio.Queue] = None, loop=None, **kw) -> None:
        super().__init__(**kw)
        self.queue = queue
        self.loop = loop

    def record(self, name: str, ms: float, extra: Optional[Dict] = None) -> Dict:
        rec = super().record(name, ms, extra)
        if self.queue is not None and self.loop is not None:
            ev = {"type": "node", "node": rec.get("node"), "ms": rec.get("ms"),
                  "extra": extra or {}}
            self.loop.call_soon_threadsafe(self.queue.put_nowait, ev)
        return rec


def _seed_corpus(vs: VectorStore, goal: str, docs_dir: Optional[str] = None) -> int:
    """摄入用户私有文档（可选）；无文档时回退内置语料，保证 mock 离线可跑。返回摄入片段数。"""
    ingested = 0
    if docs_dir:
        ingested = vs.add_dir(docs_dir)
    if not vs.docs:
        vs.add(f"关于「{goal}」的研究方法：应先界定范围，再交叉验证来源，最后结构化成文。",
               source="builtin-corpus/method")
        vs.add(f"「{goal}」的常见权衡：性能、成本与可扩展性往往相互制约，需结合场景取舍。",
               source="builtin-corpus/tradeoff")
        vs.add(f"撰写「{goal}」报告时，强制引用可溯源证据，可显著降低幻觉与编造风险。",
               source="builtin-corpus/citation")
    return ingested


def _build_and_run(goal: str, cfg: Dict, tracer: Tracer, cost: CostMeter,
                   provider: Optional[str] = None, max_iteration: Optional[int] = None,
                   out_dir: Optional[Path] = None, api_key: Optional[str] = None,
                   base_url: Optional[str] = None, docs_dir: Optional[str] = None) -> tuple[str, Dict]:
    """构建模型/Agent/图并同步运行，返回 (report, metrics)。需在独立线程内调用。"""
    if provider:
        cfg.setdefault("model", {})["provider"] = provider
    if max_iteration is not None:
        cfg.setdefault("orchestration", {})["max_iteration"] = max_iteration
    if api_key:
        cfg.setdefault("model", {})["api_key"] = api_key
    if base_url:
        cfg.setdefault("model", {})["base_url"] = base_url

    out_dir = out_dir or (ROOT / get(cfg, "output", "dir", default="outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(cfg, on_usage=cost)
    vs = VectorStore(persist_path=out_dir / "vector_store.json")
    ingested = _seed_corpus(vs, goal, docs_dir)
    web, rag, so = WebSearch(), RAGRetriever(vs), StructuredOutput(model)
    agents = {
        "researcher": Researcher(model, tools=[web, rag], config=cfg),
        "analyst": Analyst(model, tools=[so], config=cfg),
        "writer": Writer(model, tools=[], config=cfg),
        "critic": Critic(model, tools=[so], config=cfg),
    }
    ctx = WorkflowContext(model=model, agents=agents, tracer=tracer, cost=cost, config=cfg)
    g = build_graph(ctx)

    t0 = time.time()
    res = g.invoke({
        "goal": goal, "plan": [], "blackboard": {}, "draft": "",
        "critique": {}, "iteration": 0,
        "max_iteration": get(cfg, "orchestration", "max_iteration", default=3),
        "final_report": "", "trace": [],
    })
    elapsed = time.time() - t0

    report = res.get("final_report") or res.get("draft") or ""
    crit = res.get("critique", {})
    metrics = {
        "goal": goal,
        "iterations": res.get("iteration", 0),
        "passed": bool(crit.get("passed", False)),
        "score": crit.get("score"),
        "cost": cost.report(),
        "elapsed_s": round(elapsed, 2),
        "ingested": ingested,
        "trace": tracer.summary(),
    }
    return report, metrics


@app.post("/api/generate/stream")
async def generate_stream(body: GenerateBody):
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    tracer = StreamingTracer(queue=queue, loop=loop)
    cost = CostMeter(cap_cny=2.0)
    cfg = load_config()

    def run() -> None:
        try:
            report, metrics = _build_and_run(
                goal=body.goal, cfg=cfg, tracer=tracer, cost=cost,
                provider=body.provider, max_iteration=body.max_iteration,
                api_key=body.api_key, base_url=body.base_url, docs_dir=body.docs_dir,
            )
            loop.call_soon_threadsafe(
                queue.put_nowait, {"type": "done", "report": report, "metrics": metrics}
            )
        except Exception as e:  # 任何异常都保证流正常结束
            loop.call_soon_threadsafe(
                queue.put_nowait, {"type": "done", "report": f"⚠️ 生成失败：{e}", "metrics": {}}
            )

    threading.Thread(target=run, daemon=True).start()

    async def event_gen():
        while True:
            ev = await queue.get()
            yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
            if ev.get("type") == "done":
                break

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@app.post("/api/generate")
async def generate(body: GenerateBody):
    if not body.goal.strip():
        raise HTTPException(400, "goal 不能为空")
    result: Dict[str, Any] = {}
    err: list = []

    def run() -> None:
        try:
            result["report"], result["metrics"] = _build_and_run(
                goal=body.goal, cfg=load_config(), tracer=Tracer(),
                cost=CostMeter(cap_cny=2.0), provider=body.provider,
                max_iteration=body.max_iteration, api_key=body.api_key,
                base_url=body.base_url, docs_dir=body.docs_dir,
            )
        except Exception as e:
            err.append(str(e))

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join()
    if err:
        raise HTTPException(500, f"生成失败：{err[0]}")
    return JSONResponse({"report": result.get("report", ""), "metrics": result.get("metrics", {})})


@app.post("/api/export")
async def export_endpoint(body: Dict[str, Any]):
    """把报告导出为 JSON / Markdown / PDF / PPTX，作为文件下载返回。"""
    fmt = (body.get("format") or "pdf").lower()
    report = body.get("report", "")
    metrics = body.get("metrics", {})
    try:
        data, media, ext = export_report(report, metrics, fmt)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return Response(
        content=data,
        media_type=media,
        headers={"Content-Disposition": f'attachment; filename="report.{ext}"'},
    )


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse((STATIC / "index.html").read_text(encoding="utf-8"))
