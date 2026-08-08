"""模型工厂：根据 config 构建对应供应商的 ModelAdapter。"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from .adapter import ModelAdapter
from .providers.hunyuan import HunyuanProvider
from .providers.mock import MockProvider
from .providers.openai import OpenAIProvider
from .providers.qwen import QwenProvider

# 默认单价（元 / 1K token），仅用于离线成本估算；真实接入请在 config.yaml 校准。
_DEFAULT_PRICES = {
    "mock": (0.0, 0.0),
    "openai": (0.005, 0.015),
    "qwen": (0.004, 0.012),
    "hunyuan": (0.006, 0.018),
}

_DEFAULT_MODEL = {
    "mock": "mock-model",
    "openai": "gpt-4o-mini",
    "qwen": "qwen-plus",
    "hunyuan": "hunyuan-pro",
}


def build_model(cfg: Dict[str, Any], on_usage: Optional[Callable[[int, int, float], None]] = None) -> ModelAdapter:
    m = cfg.get("model", {})
    provider = m.get("provider", "mock")
    model_name = m.get("model_name") or _DEFAULT_MODEL.get(provider, "mock-model")
    temp = m.get("temperature", 0.3)
    max_tokens = m.get("max_tokens", 2048)
    pin, pout = _DEFAULT_PRICES.get(provider, (0.0, 0.0))
    price_in = m.get("price_per_1k_input", pin)
    price_out = m.get("price_per_1k_output", pout)

    common = dict(
        temperature=temp,
        max_tokens=max_tokens,
        price_per_1k_input=price_in,
        price_per_1k_output=price_out,
        on_usage=on_usage,
    )

    if provider == "mock":
        return MockProvider(model_name, **common)
    if provider == "openai":
        return OpenAIProvider(model_name, **common)
    if provider == "qwen":
        return QwenProvider(model_name, **common)
    if provider == "hunyuan":
        return HunyuanProvider(model_name, **common)
    raise ValueError(f"未知模型供应商: {provider}")
