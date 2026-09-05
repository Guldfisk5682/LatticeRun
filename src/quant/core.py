"""Reference symmetric group quantization and activation-aware clipping.

The implementation deliberately stores signed INT3 codes in int8 tensors. This
is the inspectable training/reference format; a runtime-specific 3-bit packing
kernel is a separate deployment milestone.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(slots=True)
class GroupwiseTensor:
    codes: torch.Tensor
    scales: torch.Tensor
    original_shape: tuple[int, ...]
    group_size: int
    bits: int
    clip_ratio: float
    padded_columns: int

    def dequantize(self, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        qmax = (1 << (self.bits - 1)) - 1
        if int(self.codes.min()) < -qmax or int(self.codes.max()) > qmax:
            raise ValueError("codes exceed the symmetric quantization range")
        grouped = self.codes.to(self.scales.dtype) * self.scales
        flat = grouped.reshape(self.original_shape[0], -1)
        flat = flat[:, : self.original_shape[-1]]
        result = flat.reshape(self.original_shape)
        return result.to(dtype=dtype) if dtype is not None else result


def _validate_weight(weight: torch.Tensor) -> None:
    if weight.ndim != 2:
        raise ValueError(f"expected a 2D weight, got shape {tuple(weight.shape)}")
    if not weight.is_floating_point():
        raise TypeError("weight must be floating point")


@torch.no_grad()
def quantize_symmetric_groupwise(
    weight: torch.Tensor,
    *,
    bits: int = 3,
    group_size: int = 128,
    clip_ratio: float = 1.0,
    scale_dtype: torch.dtype = torch.float16,
) -> GroupwiseTensor:
    """Quantize each output-row/input-group to [-3, 3] for INT3."""

    _validate_weight(weight)
    if not 2 <= bits <= 8:
        raise ValueError("bits must be between 2 and 8")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if not 0.0 < clip_ratio <= 1.0:
        raise ValueError("clip_ratio must be in (0, 1]")

    rows, columns = weight.shape
    padded_columns = (-columns) % group_size
    work = weight.detach().float()
    if padded_columns:
        work = F.pad(work, (0, padded_columns))
    grouped = work.reshape(rows, -1, group_size)
    qmax = (1 << (bits - 1)) - 1
    threshold = grouped.abs().amax(dim=-1, keepdim=True) * clip_ratio
    scales = threshold / qmax
    scales = torch.where(scales > 0, scales, torch.ones_like(scales))
    codes = torch.round(grouped / scales).clamp(-qmax, qmax).to(torch.int8)
    return GroupwiseTensor(
        codes=codes,
        scales=scales.to(scale_dtype),
        original_shape=tuple(weight.shape),
        group_size=group_size,
        bits=bits,
        clip_ratio=clip_ratio,
        padded_columns=padded_columns,
    )


def fake_quantize(
    weight: torch.Tensor,
    *,
    bits: int = 3,
    group_size: int = 128,
    clip_ratio: float = 1.0,
) -> torch.Tensor:
    quantized = quantize_symmetric_groupwise(
        weight,
        bits=bits,
        group_size=group_size,
        clip_ratio=clip_ratio,
        scale_dtype=torch.float32,
    )
    return quantized.dequantize(dtype=weight.dtype)


class _STEQuantize(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: object,
        weight: torch.Tensor,
        bits: int,
        group_size: int,
        clip_ratio: float,
    ) -> torch.Tensor:
        del ctx
        return fake_quantize(
            weight, bits=bits, group_size=group_size, clip_ratio=clip_ratio
        )

    @staticmethod
    def backward(
        ctx: object, gradient: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None]:
        del ctx
        return gradient, None, None, None


def ste_quantize(
    weight: torch.Tensor,
    *,
    bits: int = 3,
    group_size: int = 128,
    clip_ratio: float = 1.0,
) -> torch.Tensor:
    """Quantize in forward and use identity gradient in backward."""

    return _STEQuantize.apply(weight, bits, group_size, clip_ratio)


@torch.no_grad()
def choose_awq_clip_ratio(
    weight: torch.Tensor,
    activations: torch.Tensor,
    *,
    bits: int = 3,
    group_size: int = 128,
    ratios: Iterable[float] = (
        1.0,
        0.95,
        0.90,
        0.85,
        0.80,
        0.75,
        0.70,
        0.65,
        0.60,
        0.55,
    ),
    token_batch_size: int = 256,
) -> tuple[float, dict[float, float]]:
    """Pick clipping by activation-weighted output MSE (AWQ-style criterion)."""

    _validate_weight(weight)
    flat = activations.detach().reshape(-1, activations.shape[-1])
    if flat.shape[-1] != weight.shape[-1]:
        raise ValueError("activation hidden size does not match the weight input size")
    if flat.numel() == 0:
        raise ValueError("at least one calibration activation is required")
    reference_weight = weight.detach()
    errors: dict[float, float] = {}
    denominator = 0
    for ratio in ratios:
        ratio = float(ratio)
        candidate = fake_quantize(
            weight, bits=bits, group_size=group_size, clip_ratio=ratio
        )
        squared_error = 0.0
        elements = 0
        for start in range(0, flat.shape[0], token_batch_size):
            inputs = flat[start : start + token_batch_size].to(
                device=weight.device, dtype=weight.dtype
            )
            reference = F.linear(inputs, reference_weight)
            prediction = F.linear(inputs, candidate)
            squared_error += float(
                (prediction.float() - reference.float()).square().sum().item()
            )
            elements += prediction.numel()
        errors[ratio] = squared_error / max(elements, 1)
        denominator = max(denominator, elements)
    if denominator == 0:
        raise RuntimeError("AWQ search evaluated no activation elements")
    return min(errors, key=errors.get), errors


@torch.no_grad()
def choose_weight_mse_clip_ratio(
    weight: torch.Tensor,
    *,
    bits: int = 3,
    group_size: int = 128,
    ratios: Iterable[float] = (
        1.0,
        0.95,
        0.90,
        0.85,
        0.80,
        0.75,
        0.70,
        0.65,
        0.60,
        0.55,
    ),
) -> tuple[float, dict[float, float]]:
    """Razor-style reference grid selected by weight reconstruction MSE."""

    _validate_weight(weight)
    errors: dict[float, float] = {}
    reference = weight.detach().float()
    for ratio in ratios:
        ratio = float(ratio)
        candidate = fake_quantize(
            weight, bits=bits, group_size=group_size, clip_ratio=ratio
        )
        errors[ratio] = float((candidate.float() - reference).square().mean().item())
    return min(errors, key=errors.get), errors


@torch.no_grad()
def normalized_output_error(
    weight: torch.Tensor,
    activations: torch.Tensor,
    *,
    bits: int,
    group_size: int,
    clip_ratio: float,
    epsilon: float = 1e-12,
) -> float:
    flat = activations.detach().reshape(-1, activations.shape[-1]).to(weight)
    reference = F.linear(flat, weight)
    candidate = F.linear(
        flat,
        fake_quantize(weight, bits=bits, group_size=group_size, clip_ratio=clip_ratio),
    )
    numerator = (candidate.float() - reference.float()).square().sum()
    denominator = reference.float().square().sum().clamp_min(epsilon)
    return float((numerator / denominator).item())
