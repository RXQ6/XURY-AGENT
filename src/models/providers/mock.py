"""Mock 模型供应商 —— 无需任何 API Key 即可离线跑通整条流水线。

用途：
- 开发 / 单测 / CI 中默认使用，保证 `python main.py` 一键出报告；
- 真实接入时把 config.yaml 的 provider 改为 openai / qwen / hunyuan 即可。

Mock 通过系统提示中的 `Role: xxx` 标记识别当前角色，产出结构化/文本结果，
使 Planner/Researcher/Analyst/Critic/Writer 各司其职，端到端闭环可演示。
"""
from __future__ import annotations

import json
import re
from typing import Dict, List

from ..adapter import ModelAdapter

_ROLE_RE = re.compile(r"role:\s*([a-z_]+)", re.I)


def _detect_role(messages: List[Dict[str, str]]) -> str:
    for m in messages:
        mm = _ROLE_RE.search(m.get("content", ""))
        if mm:
            return mm.group(1).lower()
    return "unknown"


def _last_user(messages: List[Dict[str, str]]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return m.get("content", "")
    return ""


def _extract_goal(prompt: str) -> str:
    """从 prompt 中抽取真实主题（避免把整段 prompt 当成 goal）。"""
    for pat in (r"目标：([^\n]+)", r"主题：([^\n]+)"):
        m = re.search(pat, prompt)
        if m:
            return m.group(1).strip()
    return prompt[:60]


def _extract_refs(prompt: str) -> List[str]:
    """从 Writer 的 prompt 中抽取「可用引用」块，保证 mock 报告引用真实证据。"""
    m = re.search(r"可用引用：\n(.*?)(?:\n请撰写)", prompt, re.S)
    if not m:
        return []
    return [ln.strip() for ln in m.group(1).split("\n") if ln.strip()]


def _slug(s: str) -> str:
    s = re.sub(r"[^\w一-龥]+", "-", s).strip("-")
    return (s[:40] or "topic").lower()


class MockProvider(ModelAdapter):
    provider = "mock"

    def _raw_chat(self, messages, response_format=None, **kwargs):
        role = _detect_role(messages)
        prompt = _last_user(messages)
        goal = _extract_goal(prompt)
        if response_format and response_format.get("type") == "json_object":
            text = self._json_for(role, goal)
        else:
            text = self._text_for(role, goal, prompt)
        in_tok = self.count_tokens("\n".join(m["content"] for m in messages))
        out_tok = self.count_tokens(text)
        return text, in_tok, out_tok

    # ---------- JSON 角色输出 ----------
    def _json_for(self, role: str, goal: str) -> str:
        if role == "planner":
            plan = {
                "plan": [
                    {"id": "t1", "description": f"检索并取证：{goal} 的定义、背景与关键事实",
                     "role": "researcher", "depends_on": []},
                    {"id": "t2", "description": "对取证素材去重、归纳、结构化",
                     "role": "analyst", "depends_on": ["t1"]},
                    {"id": "t3", "description": "基于证据撰写带引用的深度研究报告",
                     "role": "writer", "depends_on": ["t2"]},
                ]
            }
            return json.dumps(plan, ensure_ascii=False)

        if role == "researcher":
            evidence = [
                {"title": f"{goal} — 概述与定义", "url": f"https://en.wikipedia.org/wiki/{_slug(goal)}",
                 "snippet": f"关于“{goal}”的权威定义与背景综述。"},
                {"title": f"{goal} — 技术对比分析", "url": f"https://arxiv.org/search/?query={_slug(goal)}",
                 "snippet": f"针对“{goal}”的多维度对比与实证研究结果。"},
                {"title": f"{goal} — 行业实践与案例", "url": f"https://example.com/research/{_slug(goal)}",
                 "snippet": f"“{goal}”的落地案例与最佳实践（mock 引用，仅用于离线演示）。"},
            ]
            return json.dumps({"evidence": evidence}, ensure_ascii=False)

        if role == "analyst":
            points = [
                f"核心定义：{goal} 指……（来自研究员取证，可溯源）",
                "关键维度：从性能、成本、可扩展性三个角度展开对比",
                "实证结论：现有研究表明方案间存在权衡，需结合场景取舍",
                "风险与边界：在 X / Y 条件下效果会下降，应设护栏",
            ]
            return json.dumps({"points": points}, ensure_ascii=False)

        if role == "critic":
            # 默认达标（mock Writer 产出结构完整、含引用），用于演示一次通过闭环
            return json.dumps({
                "score": 40,
                "breakdown": {"factual_accuracy": 10, "citation_completeness": 10,
                              "structure_clarity": 10, "coverage": 10},
                "feedback": "报告结构清晰、引用完整、覆盖关键维度，达到发布标准。",
                "passed": True,
            }, ensure_ascii=False)

        return json.dumps({"result": "ok"}, ensure_ascii=False)

    # ---------- 文本角色输出 ----------
    def _text_for(self, role: str, goal: str, prompt: str = "") -> str:
        if role == "writer":
            return self._mock_report(goal, prompt)
        if role == "analyst":
            return "- 要点一\n- 要点二\n- 要点三"
        return f"[mock:{role}] {goal}"

    def _mock_report(self, goal: str, prompt: str = "") -> str:
        slug = _slug(goal)
        refs = _extract_refs(prompt)
        if not refs:
            refs = [
                f"1. {goal} — 概述与定义 — https://en.wikipedia.org/wiki/{slug}",
                f"2. {goal} — 技术对比分析 — https://arxiv.org/search/?query={slug}",
                f"3. {goal} — 行业实践与案例 — https://example.com/research/{slug}",
            ]
        references = "\n".join(refs)
        return f"""# {goal}：深度研究报告

## 摘要
本报告围绕「{goal}」展开，基于可溯源的检索证据，对比核心维度并给出结论与权衡。

## 1. 背景与定义
{goal} 是当前备受关注的方向。根据权威资料，其定义与边界如下（详见引用）。

## 2. 关键维度对比
- **性能**：在典型负载下表现稳定，但高并发场景需额外优化。
- **成本**：前期投入与长期运维需综合评估。
- **可扩展性**：架构层面支持水平扩展。

## 3. 实证结论
现有研究与实践表明，各方案间存在权衡，应结合具体场景取舍。

## 4. 风险与边界
在特定条件下效果会下降，建议设置护栏与回退策略。

## 结论
{goal} 的选型应以场景匹配为先，并以可度量指标持续验证。

## References
{references}
"""
