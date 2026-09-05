"""Bounded activation capture for AWQ calibration."""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import AbstractContextManager
from typing import Self

import torch

from ..model.base import ModuleDecision


class ActivationCollector(AbstractContextManager["ActivationCollector"]):
    """Capture a bounded CPU token sample at selected Linear inputs."""

    def __init__(
        self,
        model: torch.nn.Module,
        decisions: Iterable[ModuleDecision],
        *,
        max_tokens_per_module: int = 512,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens_per_module
        self.activations: dict[str, list[torch.Tensor]] = {
            decision.name: [] for decision in decisions if decision.bits is not None
        }
        self._counts = {name: 0 for name in self.activations}
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []

    def __enter__(self) -> Self:
        modules = dict(self.model.named_modules())
        for name in self.activations:
            if name not in modules:
                raise KeyError(f"module disappeared before calibration: {name}")

            def capture(
                module: torch.nn.Module,
                args: tuple[object, ...],
                module_name: str = name,
            ) -> None:
                del module
                if not args or not isinstance(args[0], torch.Tensor):
                    return
                remaining = self.max_tokens - self._counts[module_name]
                if remaining <= 0:
                    return
                inputs = args[0].detach().reshape(-1, args[0].shape[-1])[:remaining]
                self.activations[module_name].append(
                    inputs.to(device="cpu", dtype=torch.float16)
                )
                self._counts[module_name] += inputs.shape[0]

            self._hooks.append(modules[name].register_forward_pre_hook(capture))
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        del exc_type, exc_value, traceback
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def merged(self, name: str) -> torch.Tensor:
        tensors = self.activations.get(name, [])
        if not tensors:
            raise RuntimeError(f"no calibration activations captured for {name}")
        return torch.cat(tensors, dim=0)
