"""Construction of the current quantized/recovered student."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch

from ..adapters import (
    load_adapter,
    merge_recovery_for_inference,
    prepare_quantized_student,
)
from ..config import RecoveryConfig
from ..model.base import ModelAdapter
from .common import cuda_device, load_clip_ratios


@dataclass(slots=True)
class StudentRequest:
    model: str = ""
    revision: str = "main"
    attn_implementation: str = "sdpa"
    recovery: RecoveryConfig = field(default_factory=RecoveryConfig)
    group_size: int = 128
    clip_ratios: str | Path | None = None
    adapter_checkpoint: str | Path | None = None
    enable_deltanet_z: bool = False


def load_student(
    request: StudentRequest,
    model_adapter: ModelAdapter,
    *,
    for_training: bool,
) -> torch.nn.Module:
    if not request.model:
        raise ValueError("student model_id must be provided")
    model = model_adapter.load_model(
        request.model,
        revision=request.revision,
        dtype=torch.bfloat16,
        device_map={"": cuda_device()},
        attn_implementation=request.attn_implementation,
    )
    prepare_quantized_student(
        model,
        request.recovery,
        model_adapter,
        group_size=request.group_size,
        clip_ratios=load_clip_ratios(request.clip_ratios),
        enable_deltanet_z=request.enable_deltanet_z,
    )
    if request.adapter_checkpoint:
        load_adapter(model, request.adapter_checkpoint)
    if not for_training:
        merge_recovery_for_inference(model)
    return model
