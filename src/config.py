"""Framework configuration and the validated default recipe."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

DEFAULT_MODEL_NAME = "qwen3.8-27b"
DEFAULT_AWQ_CLIP_RATIOS = (
    1.00,
    0.95,
    0.90,
    0.85,
    0.80,
    0.75,
    0.70,
    0.65,
    0.60,
    0.55,
)


@dataclass(slots=True)
class AWQConfig:
    """Activation-aware clipping search configuration."""

    clip_ratios: tuple[float, ...] = DEFAULT_AWQ_CLIP_RATIOS
    token_batch_size: int = 256
    max_calibration_tokens: int = 4096

    def __post_init__(self) -> None:
        self.clip_ratios = tuple(float(value) for value in self.clip_ratios)
        if not self.clip_ratios:
            raise ValueError("clip_ratios must not be empty")
        if any(not 0.0 < value <= 1.0 for value in self.clip_ratios):
            raise ValueError("clip ratios must be in (0, 1]")
        if self.token_batch_size <= 0 or self.max_calibration_tokens <= 0:
            raise ValueError("AWQ token limits must be positive")


@dataclass(slots=True)
class QuantConfig:
    bits: int = 3
    lm_head_bits: int = 4
    group_size: int = 128
    symmetric: bool = True
    scale_dtype: str = "float32"
    awq: AWQConfig = field(default_factory=AWQConfig)

    def __post_init__(self) -> None:
        if not 2 <= self.bits <= 8 or not 2 <= self.lm_head_bits <= 8:
            raise ValueError("weight bit widths must be between 2 and 8")
        if self.group_size <= 0:
            raise ValueError("group_size must be positive")
        if not self.symmetric:
            raise ValueError("only symmetric quantization is implemented")
        if self.scale_dtype not in {"float16", "float32"}:
            raise ValueError("scale_dtype must be float16 or float32")


@dataclass(slots=True)
class RecoveryConfig:
    adapter_type: Literal["dora", "lora"] = "dora"
    rank: int = 8
    alpha: float = 16.0
    mode: Literal["ste", "fast"] = "ste"

    @property
    def scaling(self) -> float:
        return self.alpha / self.rank

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise ValueError("adapter rank must be positive")
        if self.adapter_type == "dora" and self.mode != "ste":
            raise ValueError("DoRA currently supports merge-consistent STE only")


@dataclass(slots=True)
class LoRAConfig(RecoveryConfig):
    """Compatibility constructor for the LoRA recovery path."""

    adapter_type: Literal["lora"] = "lora"


@dataclass(slots=True)
class OPDConfig:
    max_tokens: int = 16_384
    token_chunk_size: int = 256
    temperature: float = 1.0
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    lr_scheduler_type: Literal["cosine"] = "cosine"
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    epochs: int = 1
    bf16: bool = True
    gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0
    report_to: str = "tensorboard"
    checkpoint_interval_fraction: float = 1 / 3
    save_total_limit: int = 1
    seed: int = 42
    thinking: bool = True
    reasoning_effort: Literal["low", "medium", "xhigh"] = "medium"
    do_sample: bool = True
    top_p: float = 0.95
    top_k: int = 20

    def __post_init__(self) -> None:
        if self.max_tokens <= 0 or self.token_chunk_size <= 0:
            raise ValueError("OPD token limits must be positive")
        if self.temperature <= 0 or self.learning_rate <= 0:
            raise ValueError("temperature and learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ValueError("warmup_ratio must be in [0, 1)")
        if self.train_batch_size <= 0 or self.gradient_accumulation_steps <= 0:
            raise ValueError("batch sizes must be positive")
        if self.epochs <= 0 or self.max_grad_norm <= 0:
            raise ValueError("epochs and max_grad_norm must be positive")
        if not 0.0 < self.checkpoint_interval_fraction <= 1.0:
            raise ValueError("checkpoint_interval_fraction must be in (0, 1]")
        if self.save_total_limit <= 0:
            raise ValueError("save_total_limit must be positive")
        if not 0.0 < self.top_p <= 1.0 or self.top_k < 0:
            raise ValueError("invalid rollout sampling limits")


@dataclass(slots=True)
class ProjectConfig:
    model: str = DEFAULT_MODEL_NAME
    quant: QuantConfig = field(default_factory=QuantConfig)
    recovery: RecoveryConfig = field(default_factory=RecoveryConfig)
    opd: OPDConfig = field(default_factory=OPDConfig)


def _construct_project(data: dict[str, Any]) -> ProjectConfig:
    recovery_data = dict(data.get("recovery", data.get("lora", {})))
    recovery_data.pop("dropout", None)
    quant_data = dict(data.get("quant", {}))
    awq_data = quant_data.pop("awq", {})
    return ProjectConfig(
        model=data.get("model", data.get("model_id", DEFAULT_MODEL_NAME)),
        quant=QuantConfig(**quant_data, awq=AWQConfig(**awq_data)),
        recovery=RecoveryConfig(**recovery_data),
        opd=OPDConfig(**data.get("opd", {})),
    )


def load_config(path: str | Path | None) -> ProjectConfig:
    if path is None:
        return ProjectConfig()
    with Path(path).open("r", encoding="utf-8") as handle:
        return _construct_project(json.load(handle))


def save_config(config: ProjectConfig, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2, ensure_ascii=False)
        handle.write("\n")
