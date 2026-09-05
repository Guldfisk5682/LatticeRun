"""Adapter attachment, serialization, and merge/requantization."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F

from ..config import RecoveryConfig
from ..model.base import ModelAdapter, assert_text_only_model, resolve_parent
from ..quant import fake_quantize
from .base import QuantRecoveryLinear
from .dora import QuantDoRALinear
from .lora import QuantLoRALinear


class FrozenQuantLinear(torch.nn.Module):
    def __init__(
        self,
        source: torch.nn.Linear,
        *,
        bits: int,
        group_size: int,
        clip_ratio: float,
    ) -> None:
        super().__init__()
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.bits = bits
        self.group_size = group_size
        self.clip_ratio = clip_ratio
        self.register_buffer(
            "weight",
            fake_quantize(
                source.weight.detach(),
                bits=bits,
                group_size=group_size,
                clip_ratio=clip_ratio,
            ),
        )
        self.register_buffer(
            "bias", source.bias.detach().clone() if source.bias is not None else None
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.linear(inputs, self.weight, self.bias)


class FrozenQuantEmbedding(torch.nn.Module):
    def __init__(
        self,
        source: torch.nn.Embedding,
        *,
        bits: int,
        group_size: int,
        clip_ratio: float,
    ) -> None:
        super().__init__()
        self.num_embeddings = source.num_embeddings
        self.embedding_dim = source.embedding_dim
        self.padding_idx = source.padding_idx
        self.register_buffer(
            "weight",
            fake_quantize(
                source.weight.detach(),
                bits=bits,
                group_size=group_size,
                clip_ratio=clip_ratio,
            ),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.embedding(inputs, self.weight, padding_idx=self.padding_idx)


def _recovery_layer(
    source: torch.nn.Linear,
    config: RecoveryConfig,
    *,
    bits: int,
    group_size: int,
    clip_ratio: float,
) -> QuantRecoveryLinear:
    layer_type = QuantDoRALinear if config.adapter_type == "dora" else QuantLoRALinear
    return layer_type(
        source,
        config,
        bits=bits,
        group_size=group_size,
        clip_ratio=clip_ratio,
    )


def prepare_quantized_student(
    model: torch.nn.Module,
    config: RecoveryConfig,
    model_adapter: ModelAdapter,
    *,
    group_size: int = 128,
    clip_ratios: dict[str, float] | None = None,
    enable_deltanet_z: bool = False,
) -> list[str]:
    assert_text_only_model(model)
    decisions = model_adapter.inspect_named_modules(
        model.named_modules(), enable_deltanet_z=enable_deltanet_z, strict=True
    )
    replaced: list[str] = []
    for decision in decisions:
        if decision.bits is None:
            continue
        source = model.get_submodule(decision.name)
        ratio = (clip_ratios or {}).get(decision.name, 1.0)
        if decision.opd_eligible:
            if not isinstance(source, torch.nn.Linear):
                raise TypeError(f"OPD target is not Linear: {decision.name}")
            replacement: torch.nn.Module = _recovery_layer(
                source,
                config,
                bits=decision.bits,
                group_size=group_size,
                clip_ratio=ratio,
            )
            replaced.append(decision.name)
        elif isinstance(source, torch.nn.Linear):
            replacement = FrozenQuantLinear(
                source,
                bits=decision.bits,
                group_size=group_size,
                clip_ratio=ratio,
            )
        elif isinstance(source, torch.nn.Embedding):
            replacement = FrozenQuantEmbedding(
                source,
                bits=decision.bits,
                group_size=group_size,
                clip_ratio=ratio,
            )
        else:
            raise TypeError(
                f"quantization target type unsupported: {decision.name}: {type(source)}"
            )
        parent, leaf = resolve_parent(model, decision.name)
        setattr(parent, leaf, replacement)
    if not replaced:
        raise RuntimeError("no OPD-eligible modules were found")
    return replaced


def trainable_adapter_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    parameters: list[torch.nn.Parameter] = []
    for module in model.modules():
        if isinstance(module, QuantRecoveryLinear):
            for parameter in module.adapter_parameters():
                parameter.requires_grad_(True)
                parameters.append(parameter)
    return parameters


def save_adapter(
    model: torch.nn.Module,
    destination: str | Path,
    config: RecoveryConfig,
) -> None:
    from safetensors.torch import save_file

    target = Path(destination)
    target.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, torch.Tensor] = {}
    modules: dict[str, dict[str, object]] = {}
    for name, module in model.named_modules():
        if not isinstance(module, QuantRecoveryLinear):
            continue
        tensors[f"{name}.lora_a"] = module.lora_a.detach().cpu().contiguous()
        tensors[f"{name}.lora_b"] = module.lora_b.detach().cpu().contiguous()
        if isinstance(module, QuantDoRALinear):
            tensors[f"{name}.magnitude"] = module.magnitude.detach().cpu().contiguous()
        modules[name] = {
            "adapter_type": module.adapter_type,
            "bits": module.bits,
            "group_size": module.group_size,
            "clip_ratio": module.clip_ratio,
        }
    save_file(tensors, target / "adapter.safetensors")
    with (target / "adapter_config.json").open("w", encoding="utf-8") as handle:
        json.dump({"recovery": asdict(config), "modules": modules}, handle, indent=2)
        handle.write("\n")


def load_adapter(model: torch.nn.Module, source: str | Path) -> None:
    from safetensors.torch import load_file

    source_path = Path(source)
    with (source_path / "adapter_config.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    recovery_metadata = metadata.get("recovery")
    if recovery_metadata is None:
        if "lora" not in metadata:
            raise KeyError("adapter metadata has neither 'recovery' nor legacy 'lora'")
        expected_type = "lora"
    else:
        expected_type = recovery_metadata["adapter_type"]
    tensors = load_file(source_path / "adapter.safetensors")
    used: set[str] = set()
    for name, module in model.named_modules():
        if not isinstance(module, QuantRecoveryLinear):
            continue
        if module.adapter_type != expected_type:
            raise ValueError(
                f"adapter type mismatch: checkpoint={expected_type}, model={module.adapter_type}"
            )
        keys = [f"{name}.lora_a", f"{name}.lora_b"]
        if isinstance(module, QuantDoRALinear):
            keys.append(f"{name}.magnitude")
        if any(key not in tensors for key in keys):
            raise KeyError(f"adapter is missing tensors for {name}")
        module.lora_a.data.copy_(tensors[keys[0]].to(module.lora_a))
        module.lora_b.data.copy_(tensors[keys[1]].to(module.lora_b))
        if isinstance(module, QuantDoRALinear):
            module.magnitude.data.copy_(tensors[keys[2]].to(module.magnitude))
        used.update(keys)
    unexpected = set(tensors) - used
    if unexpected:
        raise KeyError(f"adapter has unexpected tensors: {sorted(unexpected)[:5]}")


@torch.no_grad()
def merge_recovery_for_inference(model: torch.nn.Module) -> list[str]:
    """Materialize, requantize, and keep every replacement on execution device."""

    target_names = [
        name
        for name, module in model.named_modules()
        if isinstance(module, QuantRecoveryLinear)
    ]
    for name in target_names:
        module = model.get_submodule(name)
        if not isinstance(module, QuantRecoveryLinear):
            raise TypeError(f"recovery target changed during merge: {name}")
        device = module.execution_device
        effective = module.effective_weight().to(device=device)
        merged = torch.nn.Linear(
            module.in_features,
            module.out_features,
            bias=module.base_bias is not None,
            device=device,
            dtype=effective.dtype,
        )
        merged.weight.copy_(effective)
        if module.base_bias is not None:
            merged.bias.copy_(module.base_bias.to(device=device, dtype=effective.dtype))
        replacement = FrozenQuantLinear(
            merged,
            bits=module.bits,
            group_size=module.group_size,
            clip_ratio=module.clip_ratio,
        )
        parent, leaf = resolve_parent(model, name)
        setattr(parent, leaf, replacement)
        del effective, merged, module
    return target_names
