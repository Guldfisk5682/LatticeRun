"""Model and adapter registry used by public orchestration boundaries."""

from __future__ import annotations

from .base import ModelAdapter, ModelSpec
from .qwen35 import QWEN35_ADAPTER, QWEN38_MODEL_ID

QWEN38_SPEC = ModelSpec(
    name="qwen3.8-27b",
    model_id=QWEN38_MODEL_ID,
    adapter=QWEN35_ADAPTER.name,
)

_MODEL_ADAPTERS: dict[str, ModelAdapter] = {QWEN35_ADAPTER.name: QWEN35_ADAPTER}
_MODEL_SPECS: dict[str, ModelSpec] = {QWEN38_SPEC.name: QWEN38_SPEC}


def available_model_adapters() -> tuple[str, ...]:
    return tuple(sorted(_MODEL_ADAPTERS))


def available_models() -> tuple[str, ...]:
    return tuple(sorted(_MODEL_SPECS))


def register_model_adapter(adapter: ModelAdapter, *, replace: bool = False) -> None:
    if not adapter.name:
        raise ValueError("model adapter name must be non-empty")
    if adapter.name in _MODEL_ADAPTERS and not replace:
        raise ValueError(f"model adapter is already registered: {adapter.name}")
    _MODEL_ADAPTERS[adapter.name] = adapter


def register_model(spec: ModelSpec, *, replace: bool = False) -> None:
    if not spec.name or not spec.model_id or not spec.adapter:
        raise ValueError("model specs require name, model_id, and adapter")
    if spec.name in _MODEL_SPECS and not replace:
        raise ValueError(f"model is already registered: {spec.name}")
    _MODEL_SPECS[spec.name] = spec


def resolve_model_adapter(name: str) -> ModelAdapter:
    try:
        return _MODEL_ADAPTERS[name]
    except KeyError as error:
        available = ", ".join(available_model_adapters())
        raise ValueError(
            f"unknown model adapter {name!r}; available: {available}"
        ) from error


def resolve_model(name_or_id: str, adapter: str | None = None) -> tuple[str, ModelAdapter]:
    """Resolve a registered name, or an explicit Hub ID plus adapter name."""

    spec = _MODEL_SPECS.get(name_or_id)
    if spec is not None:
        if adapter is not None and adapter != spec.adapter:
            raise ValueError(
                f"registered model {name_or_id!r} requires adapter {spec.adapter!r}"
            )
        return spec.model_id, resolve_model_adapter(spec.adapter)
    if adapter is None:
        raise ValueError("unregistered model IDs require an explicit model adapter")
    return name_or_id, resolve_model_adapter(adapter)
