"""LoRA recovery paths: merge-consistent STE and parallel FAST baseline."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..config import RecoveryConfig
from ..quant import fake_quantize, ste_quantize
from .base import QuantRecoveryLinear


class QuantLoRALinear(QuantRecoveryLinear):
    adapter_type = "lora"

    def __init__(
        self,
        source: torch.nn.Linear,
        config: RecoveryConfig,
        *,
        bits: int,
        group_size: int,
        clip_ratio: float,
    ) -> None:
        super().__init__(
            source,
            config,
            bits=bits,
            group_size=group_size,
            clip_ratio=clip_ratio,
        )
        if self.mode == "fast":
            self.register_buffer(
                "fast_base_weight",
                fake_quantize(
                    source.weight.detach(),
                    bits=bits,
                    group_size=group_size,
                    clip_ratio=clip_ratio,
                ),
                persistent=False,
            )
        else:
            self.register_buffer("fast_base_weight", None, persistent=False)

    def effective_weight(self) -> torch.Tensor:
        return self.base_on_execution_device().detach() + self.delta_weight()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.mode == "fast":
            base = F.linear(inputs, self.fast_base_weight, self.base_bias)
            update = F.linear(F.linear(inputs, self.lora_a), self.lora_b)
            return base + update * self.scaling
        quantized = ste_quantize(
            self.effective_weight(),
            bits=self.bits,
            group_size=self.group_size,
            clip_ratio=self.clip_ratio,
        )
        return F.linear(inputs, quantized, self.base_bias)
