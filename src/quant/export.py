"""Sharded, inspectable low-bit checkpoint export."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from ..model.base import ModuleDecision, assert_text_only_model
from .core import quantize_symmetric_groupwise


def export_quantized_checkpoint(
    model: torch.nn.Module,
    decisions: list[ModuleDecision],
    destination: str | Path,
    *,
    group_size: int,
    clip_ratios: dict[str, float] | None = None,
    max_shard_bytes: int = 3_500_000_000,
) -> Path:
    """Export signed codes/scales plus protected tensors as safetensor shards."""

    from safetensors.torch import save_file

    assert_text_only_model(model)
    target = Path(destination)
    target.mkdir(parents=True, exist_ok=True)
    decision_map = {item.name: item for item in decisions}
    modules = dict(model.named_modules())
    output: dict[str, torch.Tensor] = {}
    output_bytes = 0
    shard_index = 1
    weight_map: dict[str, str] = {}
    quant_metadata: dict[str, dict[str, object]] = {}

    def flush() -> None:
        nonlocal output, output_bytes, shard_index
        if not output:
            return
        filename = f"model-{shard_index:05d}.safetensors"
        save_file(output, target / filename)
        for key in output:
            weight_map[key] = filename
        output = {}
        output_bytes = 0
        shard_index += 1

    consumed: set[str] = set()
    for module_name, decision in decision_map.items():
        module = modules[module_name]
        weight = getattr(module, "weight", None)
        if weight is None or decision.bits is None:
            continue
        quantized = quantize_symmetric_groupwise(
            weight,
            bits=decision.bits,
            group_size=group_size,
            clip_ratio=(clip_ratios or {}).get(module_name, 1.0),
            scale_dtype=torch.float32,
        )
        prefix = f"{module_name}.weight"
        tensors = {
            f"{prefix}.codes": quantized.codes.cpu().contiguous(),
            f"{prefix}.scales": quantized.scales.cpu().contiguous(),
        }
        tensor_bytes = sum(
            item.numel() * item.element_size() for item in tensors.values()
        )
        if output and output_bytes + tensor_bytes > max_shard_bytes:
            flush()
        output.update(tensors)
        output_bytes += tensor_bytes
        consumed.add(prefix)
        quant_metadata[prefix] = {
            "bits": decision.bits,
            "group_size": group_size,
            "clip_ratio": quantized.clip_ratio,
            "shape": list(quantized.original_shape),
            "padded_columns": quantized.padded_columns,
            "codes_dtype": "int8",
            "scales_dtype": "float32",
        }

    for name, tensor in model.state_dict().items():
        if name in consumed:
            continue
        cpu = tensor.detach().cpu().contiguous()
        tensor_bytes = cpu.numel() * cpu.element_size()
        if output and output_bytes + tensor_bytes > max_shard_bytes:
            flush()
        output[name] = cpu
        output_bytes += tensor_bytes
    flush()
    index = {
        "format": "latticerun-reference-v1",
        "note": "INT3 codes use int8 storage; dense 3-bit runtime packing is not claimed",
        "weight_map": weight_map,
        "quantization": quant_metadata,
    }
    index_path = target / "model.latticerun.index.json"
    with index_path.open("w", encoding="utf-8") as handle:
        json.dump(index, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return index_path
