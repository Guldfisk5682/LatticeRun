"""PEFT-compatible row-wise DoRA materialization with STE quantization."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..config import RecoveryConfig
from ..quant import ste_quantize
from .base import QuantRecoveryLinear


class QuantDoRALinear(QuantRecoveryLinear):
    """DoRA direction update plus independently trainable row magnitude.

    Like the validated PEFT DoRA formulation, the dynamic direction norm is
    detached from autograd. For a PyTorch Linear weight shaped ``[out, in]``,
    magnitudes and norms are row-wise over ``in``.
    """

    adapter_type = "dora"

    def __init__(
        self,
        source: torch.nn.Linear,
        config: RecoveryConfig,
        *,
        bits: int,
        group_size: int,
        clip_ratio: float,
    ) -> None:
        if config.mode != "ste":
            raise ValueError("DoRA is implemented only for merge-consistent STE")
        super().__init__(
            source,
            config,
            bits=bits,
            group_size=group_size,
            clip_ratio=clip_ratio,
        )
        initial_magnitude = torch.linalg.vector_norm(
            source.weight.detach().float(), dim=1
        ).to(device=source.weight.device, dtype=source.weight.dtype)
        self.magnitude = torch.nn.Parameter(initial_magnitude)

    def effective_weight(self) -> torch.Tensor:
        direction = self.base_on_execution_device().detach() + self.delta_weight()
        direction_norm = (
            torch.linalg.vector_norm(direction.float(), dim=1, keepdim=True)
            .clamp_min(torch.finfo(torch.float32).eps)
            .detach()
            .to(direction.dtype)
        )
        magnitude = self.magnitude.to(direction.dtype).unsqueeze(1)
        return direction * (magnitude / direction_norm)

    def adapter_parameters(self) -> list[torch.nn.Parameter]:
        return [self.lora_a, self.lora_b, self.magnitude]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        quantized = ste_quantize(
            self.effective_weight(),
            bits=self.bits,
            group_size=self.group_size,
            clip_ratio=self.clip_ratio,
        )
        return F.linear(inputs, quantized, self.base_bias)
