"""Merge-consistent recovery export as auditable low-bit codes and scales."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from ..adapters import QuantDoRALinear, QuantLoRALinear
from ..config import RecoveryConfig
from ..model.base import ModelAdapter, assert_text_only_model
from ..quant import quantize_symmetric_groupwise
from .common import cuda_device, load_clip_ratios
from .student import StudentRequest


@dataclass(slots=True)
class ReferenceExportRequest:
    """Export parameters for the post-OPD semantic low-bit checkpoint."""

    student: StudentRequest = field(default_factory=StudentRequest)
    output: str | Path = "reference-student"
    max_shard_bytes: int = 3_500_000_000


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_adapter_payload(
    source: str | Path, expected: RecoveryConfig
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    source_path = Path(source)
    metadata = json.loads(
        (source_path / "adapter_config.json").read_text(encoding="utf-8")
    )
    recovery = metadata.get("recovery")
    if recovery is None:
        raise RuntimeError("reference export requires current recovery metadata")
    stored = RecoveryConfig(**recovery)
    if stored != expected:
        raise RuntimeError(
            f"adapter recovery config mismatch: stored={stored}, requested={expected}"
        )
    return load_file(source_path / "adapter.safetensors"), metadata


@torch.no_grad()
def export_reference_student(
    request: ReferenceExportRequest, model_adapter: ModelAdapter
) -> dict[str, int | float | str]:
    """Export the evaluated merged/requantized student as codes plus FP32 scales.

    Codes intentionally use inspectable int8 storage. This is the exact semantic
    low-bit source for the later dense INT3 packer, not yet a packed deployment
    checkpoint and not a low-VRAM runtime claim.
    """

    if request.student.adapter_checkpoint is None:
        raise ValueError("post-OPD reference export requires --adapter")
    if request.max_shard_bytes <= 0:
        raise ValueError("max_shard_bytes must be positive")
    target = Path(request.output)
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"reference export target is not empty: {target}")
    target.mkdir(parents=True, exist_ok=True)
    incomplete = target / "INCOMPLETE"
    incomplete.write_text("export in progress\n", encoding="utf-8")

    started = time.perf_counter()
    model = model_adapter.load_model(
        request.student.model,
        revision=request.student.revision,
        dtype=torch.bfloat16,
        device_map={"": cuda_device()},
        attn_implementation=request.student.attn_implementation,
    )
    model.eval()
    assert_text_only_model(model)
    decisions = model_adapter.inspect_named_modules(
        model.named_modules(),
        enable_deltanet_z=request.student.enable_deltanet_z,
        strict=True,
    )
    modules = dict(model.named_modules())
    ratios = load_clip_ratios(request.student.clip_ratios)
    adapter_tensors, adapter_metadata = _load_adapter_payload(
        request.student.adapter_checkpoint, request.student.recovery
    )
    expected_modules = {
        decision.name for decision in decisions if decision.opd_eligible
    }
    metadata_modules = set((adapter_metadata.get("modules") or {}).keys())
    if metadata_modules != expected_modules:
        raise RuntimeError(
            "adapter module metadata does not match the active model policy"
        )
    expected_adapter_keys = {
        f"{name}.{suffix}"
        for name in expected_modules
        for suffix in (
            ("lora_a", "lora_b", "magnitude")
            if request.student.recovery.adapter_type == "dora"
            else ("lora_a", "lora_b")
        )
    }
    missing = expected_adapter_keys - set(adapter_tensors)
    unexpected = set(adapter_tensors) - expected_adapter_keys
    if missing:
        raise KeyError(f"adapter is missing tensors: {sorted(missing)[:5]}")
    if unexpected:
        raise KeyError(f"adapter has unexpected tensors: {sorted(unexpected)[:5]}")

    output: dict[str, torch.Tensor] = {}
    output_bytes = 0
    shard_index = 1
    weight_map: dict[str, str] = {}
    quantization: dict[str, dict[str, object]] = {}
    consumed: set[str] = set()

    def flush() -> None:
        nonlocal output, output_bytes, shard_index
        if not output:
            return
        filename = f"model-{shard_index:05d}.safetensors"
        save_file(output, target / filename)
        weight_map.update({key: filename for key in output})
        output = {}
        output_bytes = 0
        shard_index += 1

    for decision in decisions:
        if decision.bits is None:
            continue
        source = modules[decision.name]
        clip_ratio = ratios.get(decision.name, 1.0)
        if decision.opd_eligible:
            if not isinstance(source, torch.nn.Linear):
                raise TypeError(f"OPD export target is not Linear: {decision.name}")
            layer_type = (
                QuantDoRALinear
                if request.student.recovery.adapter_type == "dora"
                else QuantLoRALinear
            )
            recovery_layer = layer_type(
                source,
                request.student.recovery,
                bits=decision.bits,
                group_size=request.student.group_size,
                clip_ratio=clip_ratio,
            )
            keys = [f"{decision.name}.lora_a", f"{decision.name}.lora_b"]
            recovery_layer.lora_a.copy_(
                adapter_tensors[keys[0]].to(recovery_layer.lora_a)
            )
            recovery_layer.lora_b.copy_(
                adapter_tensors[keys[1]].to(recovery_layer.lora_b)
            )
            if isinstance(recovery_layer, QuantDoRALinear):
                magnitude_key = f"{decision.name}.magnitude"
                recovery_layer.magnitude.copy_(
                    adapter_tensors[magnitude_key].to(recovery_layer.magnitude)
                )
                keys.append(magnitude_key)
            effective_weight = recovery_layer.effective_weight()
        elif isinstance(source, (torch.nn.Linear, torch.nn.Embedding)):
            recovery_layer = None
            effective_weight = source.weight
        else:
            raise TypeError(
                f"quantized reference target is unsupported: {decision.name}: {type(source)}"
            )

        quantized = quantize_symmetric_groupwise(
            effective_weight,
            bits=decision.bits,
            group_size=request.student.group_size,
            clip_ratio=clip_ratio,
            scale_dtype=torch.float32,
        )
        prefix = f"{decision.name}.weight"
        tensors = {
            f"{prefix}.codes": quantized.codes.cpu().contiguous(),
            f"{prefix}.scales": quantized.scales.cpu().contiguous(),
        }
        tensor_bytes = sum(
            tensor.numel() * tensor.element_size() for tensor in tensors.values()
        )
        if output and output_bytes + tensor_bytes > request.max_shard_bytes:
            flush()
        output.update(tensors)
        output_bytes += tensor_bytes
        consumed.add(prefix)
        quantization[prefix] = {
            "bits": decision.bits,
            "group_size": request.student.group_size,
            "clip_ratio": clip_ratio,
            "shape": list(quantized.original_shape),
            "padded_columns": quantized.padded_columns,
            "codes_dtype": "int8",
            "scales_dtype": "float32",
            "opd_merged": decision.opd_eligible,
        }
        del tensors, quantized, effective_weight, recovery_layer

    for name, tensor in model.state_dict().items():
        if name in consumed:
            continue
        cpu = tensor.detach().cpu().contiguous()
        tensor_bytes = cpu.numel() * cpu.element_size()
        if output and output_bytes + tensor_bytes > request.max_shard_bytes:
            flush()
        output[name] = cpu
        output_bytes += tensor_bytes
    flush()

    model.config.save_pretrained(target)
    tokenizer = AutoTokenizer.from_pretrained(
        request.student.model, revision=request.student.revision
    )
    tokenizer.save_pretrained(target)
    index = {
        "format": "latticerun-effective-reference-v1",
        "note": "Semantic INT3/INT4 codes use int8 storage; dense 3-bit packing is the next runtime step",
        "source_model": request.student.model,
        "source_revision": request.student.revision,
        "adapter_checkpoint": str(request.student.adapter_checkpoint),
        "recovery": {
            "adapter_type": request.student.recovery.adapter_type,
            "mode": request.student.recovery.mode,
            "rank": request.student.recovery.rank,
            "alpha": request.student.recovery.alpha,
        },
        "packed_int3": False,
        "low_vram_claim": False,
        "weight_map": weight_map,
        "quantization": quantization,
    }
    _atomic_json(target / "model.latticerun.index.json", index)
    incomplete.unlink()
    size_bytes = sum(
        path.stat().st_size for path in target.rglob("*") if path.is_file()
    )
    return {
        "output": str(target),
        "quantized_modules": len(quantization),
        "size_bytes": size_bytes,
        "wall_seconds": time.perf_counter() - started,
    }
