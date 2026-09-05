"""Generic model-adapter contracts and module decisions."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, Protocol

import torch

ReasoningEffort = Literal["low", "medium", "xhigh"]


@dataclass(frozen=True, slots=True)
class ModuleDecision:
    name: str
    role: str
    bits: int | None
    compute_dtype: str
    opd_eligible: bool
    reason: str


class ModelAdapter(Protocol):
    """Minimal model-specific surface used by generic project workflows."""

    name: str

    def load_model(
        self,
        model_id: str,
        *,
        revision: str,
        dtype: torch.dtype,
        device_map: str | dict[str, object] | None,
        attn_implementation: str | None,
    ) -> torch.nn.Module: ...

    def inspect_named_modules(
        self,
        named_modules: Iterable[tuple[str, object]],
        *,
        enable_deltanet_z: bool = False,
        strict: bool = True,
    ) -> list[ModuleDecision]: ...

    def render_prompt(
        self,
        tokenizer: object,
        row: dict[str, object],
        *,
        thinking: bool,
        reasoning_effort: ReasoningEffort | None,
        force_chat_template: bool,
    ) -> str: ...

    def block_name_for_module(self, module_name: str) -> str:
        """Return the enclosing decoder-block module for a leaf module."""
        ...

    def forward_hidden(
        self, model: torch.nn.Module, **model_inputs: torch.Tensor
    ) -> torch.Tensor:
        """Run the language backbone and return its last hidden states."""
        ...


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """A public model name resolved to a Hub model and an adapter."""

    name: str
    model_id: str
    adapter: str


def assert_text_only_model(model: torch.nn.Module) -> None:
    forbidden = [
        name
        for name, _ in model.named_modules()
        if name and ("visual" in name.lower() or "vision" in name.lower())
    ]
    if forbidden:
        preview = ", ".join(forbidden[:5])
        raise RuntimeError(f"vision tower is forbidden in quantization/OPD: {preview}")


def resolve_parent(
    model: torch.nn.Module, qualified_name: str
) -> tuple[torch.nn.Module, str]:
    parts = qualified_name.split(".")
    parent = model
    for part in parts[:-1]:
        if part.isdigit() and isinstance(
            parent, (torch.nn.ModuleList, torch.nn.Sequential)
        ):
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    return parent, parts[-1]
