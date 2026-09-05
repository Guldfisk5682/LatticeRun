"""Model contracts with lazily loaded concrete adapters."""

from __future__ import annotations

from .base import (
    ModelAdapter,
    ModelSpec,
    ModuleDecision,
    assert_text_only_model,
    resolve_parent,
)

_QWEN_EXPORTS = {
    "QWEN35_ADAPTER",
    "QWEN38_MODEL_ID",
    "QwenRole",
    "Qwen35Adapter",
    "classify_module",
    "decide_module",
    "force_deltanet_control_fp32",
    "inspect_named_modules",
    "load_text_config",
    "load_text_only_causal_lm",
    "validate_qwen38_text_config",
}
_REGISTRY_EXPORTS = {
    "available_model_adapters",
    "available_models",
    "register_model",
    "register_model_adapter",
    "resolve_model",
    "resolve_model_adapter",
}


def __getattr__(name: str):
    if name in _QWEN_EXPORTS:
        from . import qwen35

        return getattr(qwen35, name)
    if name in _REGISTRY_EXPORTS:
        from . import registry

        return getattr(registry, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "QWEN35_ADAPTER",
    "QWEN38_MODEL_ID",
    "ModelAdapter",
    "ModelSpec",
    "ModuleDecision",
    "Qwen35Adapter",
    "QwenRole",
    "assert_text_only_model",
    "available_model_adapters",
    "available_models",
    "classify_module",
    "decide_module",
    "force_deltanet_control_fp32",
    "inspect_named_modules",
    "load_text_config",
    "load_text_only_causal_lm",
    "register_model",
    "register_model_adapter",
    "resolve_model",
    "resolve_model_adapter",
    "resolve_parent",
    "validate_qwen38_text_config",
]
