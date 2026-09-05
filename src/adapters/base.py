"""Common recovery-adapter interface and storage policy."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch

from ..config import RecoveryConfig
from ..quant import GroupwiseTensor, quantize_symmetric_groupwise


class QuantRecoveryLinear(torch.nn.Module, ABC):
    adapter_type: str

    def __init__(
        self,
        source: torch.nn.Linear,
        config: RecoveryConfig,
        *,
        bits: int,
        group_size: int,
        clip_ratio: float,
    ) -> None:
        super().__init__()
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.rank = config.rank
        self.scaling = config.scaling
        self.mode = config.mode
        self.bits = bits
        self.group_size = group_size
        self.clip_ratio = clip_ratio
        base_device = "cpu" if config.mode == "fast" else source.weight.device
        self.register_buffer(
            "base_weight",
            source.weight.detach().to(device=base_device, copy=True),
            persistent=True,
        )
        self.register_buffer(
            "base_bias",
            source.bias.detach().clone() if source.bias is not None else None,
            persistent=True,
        )
        self.lora_a = torch.nn.Parameter(
            torch.empty(
                config.rank,
                source.in_features,
                device=source.weight.device,
                dtype=source.weight.dtype,
            )
        )
        self.lora_b = torch.nn.Parameter(
            torch.zeros(
                source.out_features,
                config.rank,
                device=source.weight.device,
                dtype=source.weight.dtype,
            )
        )
        torch.nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    @property
    def execution_device(self) -> torch.device:
        return self.lora_a.device

    def base_on_execution_device(self) -> torch.Tensor:
        return self.base_weight.to(
            device=self.execution_device, dtype=self.lora_a.dtype, non_blocking=True
        )

    def delta_weight(self) -> torch.Tensor:
        return (self.lora_b @ self.lora_a) * self.scaling

    @abstractmethod
    def effective_weight(self) -> torch.Tensor:
        """Materialize the ordinary dense weight before requantization."""

    def adapter_parameters(self) -> list[torch.nn.Parameter]:
        return [self.lora_a, self.lora_b]

    @torch.no_grad()
    def merged_quantized(self) -> GroupwiseTensor:
        return quantize_symmetric_groupwise(
            self.effective_weight(),
            bits=self.bits,
            group_size=self.group_size,
            clip_ratio=self.clip_ratio,
        )
