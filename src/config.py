"""配置加载：config.yaml + .env（环境变量优先级更高）。"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv

load_dotenv()

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def load_config(path: str | os.PathLike | None = None) -> Dict[str, Any]:
    """加载配置，并用相关环境变量覆盖。"""
    p = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # 环境变量覆盖
    prov = os.getenv("MODEL_PROVIDER")
    if prov:
        cfg.setdefault("model", {})["provider"] = prov

    budget = os.getenv("BUDGET_CAP_CNY")
    if budget:
        cfg.setdefault("orchestration", {})["budget_cap_cny"] = float(budget)

    return cfg


def get(cfg: Dict[str, Any], *keys, default=None):
    """安全取值：get(cfg, 'a', 'b') 等效 cfg['a']['b']。"""
    cur: Any = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur
