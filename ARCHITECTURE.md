# 架构与设计映射

本文档将 `设计报告.md` 的各部分映射到本仓库的实现，便于评审与面试讲解。

## 1. 项目概述 / 需求（报告 §1–§2）

- **FR-1~FR-9** 全部落地：`main.py` 实现主题→报告；`graph/builder.py` 的 `planner_node` 做任务分解（DAG），`dispatcher_node` 拓扑派发；研究员取证、分析员结构化、撰写员成文、评审员打分；`observability/` 记录 token 成本、耗时、迭代。
- **非功能需求**：可扩展性（新增角色只需实现 `BaseAgent`）、成本可控（`config.yaml` 可切模型 + `budget_cap_cny`）、防死循环（`max_iteration`）、可度量（trace/cost/eval）。

## 2. 总体架构（报告 §3）

分层架构与报告一致：接入层 / 编排层(LangGraph) / 角色层 / 工具层 / 记忆层 / 模型适配层 / 可观测评测层。
技术选型：**编排用 LangGraph**；Agent / Tool / 质量门 / 评测自研；模型用统一 Adapter；记忆用 Blackboard + 向量库。

## 3. 模块详细设计（报告 §4）

| 设计项 | 实现位置 |
|--------|----------|
| 编排层节点/边/质量门 | `src/graph/builder.py`（`_planner/_dispatcher/_researcher/_analyst/_writer/_critic` + `_quality_gate`） |
| 角色层 4 类 Agent | `src/agents/{researcher,analyst,writer,critic}.py`，继承 `src/agents/base.py` |
| 工具层统一接口 | `src/tools/base.py`（`Tool` 抽象）；`web_search.py` / `rag.py` |
| 记忆层 | `src/memory/blackboard.py`（Blackboard）、`vector_store.py`（TF 向量 + 余弦，可持久化） |
| 模型适配层 | `src/models/adapter.py`（`ModelAdapter`：`chat/complete/count_tokens/on_usage`）+ `providers/{mock,openai,qwen,hunyuan}.py` |
| 质量门 / Critic | `src/agents/critic.py` + `config.yaml` 的 `quality_threshold` / `max_iteration` |
| 可观测 / 评测 | `src/observability/{tracer,cost}.py`、`eval/run_eval.py` |

## 4. 关键接口与数据结构（报告 §5）

- **`ReportState`**（`src/graph/state.py`）：与报告 TypedDict 一致。`blackboard` 用 merge reducer 合并各节点写入；`iteration` 用 add reducer 累加；`trace` 用 list add 累积。
- **`Tool` 协议**（`src/tools/base.py`）：`name/description/input_schema/execute`。
- **`BaseAgent`**（`src/agents/base.py`）：`role/system_prompt/tools/run(task, state)`；每个 Agent 读取 `state['blackboard']` 作为上下文，返回本角色产物，写回由图节点负责（状态更新可追踪、可合并）。

```python
class ReportState(TypedDict):
    goal: str
    plan: List[SubTask]
    blackboard: Annotated[Dict[str, Any], _merge_dict]   # 共享产物
    draft: str
    critique: Dict[str, Any]                             # {score, breakdown, feedback, passed}
    iteration: Annotated[int, operator.add]
    max_iteration: int
    final_report: str
    trace: Annotated[List[Dict], operator.add]
```

## 5. 数据流转与算法（报告 §6）

端到端步骤对应 `builder.py` 的节点链：
1. `planner_node`：LLM 将 goal 拆为 `plan`（子任务 + 依赖 + 角色）。
2. `dispatcher_node`：拓扑排序，写入 `blackboard['schedule']`。
3. `researcher_node` / `analyst_node`：写 `evidence` / `analysis` 到 blackboard。
4. `writer_node`：基于 blackboard 成文 → `draft`，`iteration += 1`。
5. `critic_node`：打分 → `critique`；达标或达上限则写 `final_report`。
6. 质量门条件边：达标→END；否则回流 `writer`。

迭代终止条件：`critique.passed == True` **或** `iteration >= max_iteration`。

## 6. 工程结构 / 里程碑（报告 §7–§8）

目录结构与报告 §7 一致。实现按阶段推进：脚手架 → 角色 → 质量门 → 可观测/评测 → 收尾。

## 7. 评测 / 风险（报告 §9–§10）

- 评测集 `eval/cases.json`（12 例），`run_eval.py` 输出通过率/成本/耗时报表。
- 风险对策均已落地：预算上限（`budget_cap_cny` + `CostMeter` 超支告警）、结构化输出 + 失败重试（JSON 容错解析）、`max_iteration` 防死循环、Writer 禁无引用 + Critic 校验、RAG 先接本地语料再放开 Web。

## 8. 实现说明与偏差

- **`main.py` 位置**：CLI 入口同时存在于 `main.py`（根，兼容 `python main.py`）与 `src/main.py`（模块，兼容 `python -m src.main`）。
- **并行度**：Dispatcher 产出 DAG 拓扑序；当前参考实现按拓扑顺序执行（研究员→分析员存在依赖），保证正确性；架构支持无依赖子任务并行。
- **离线可运行**：默认 `provider=mock`（`src/models/providers/mock.py`）使整条流水线无需 API Key 即可跑通与单测；接入真实模型仅改配置。
- **向量库**：默认内置轻量向量库（零依赖）；`requirements.txt` 备注了可选 `chromadb`，接口保持一致，生产可替换。
