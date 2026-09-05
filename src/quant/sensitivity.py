"""Target-only decoder-block output sensitivity measurements."""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from ..adapters import FrozenQuantLinear
from ..model.base import ModelAdapter, resolve_parent
from ..opd.common import cuda_device, read_jsonl


@dataclass(frozen=True, slots=True)
class BlockSensitivityTarget:
    name: str
    local_error: float
    clip_ratio: float
    selection: str


@dataclass(slots=True)
class BlockSensitivityRequest:
    model: str = ""
    revision: str = "main"
    attn_implementation: str = "sdpa"
    calibration_jsonl: str | Path = "calibration.jsonl"
    awq_diagnostics: str | Path = "awq_diagnostics.json"
    output: str | Path = "block_sensitivity.json"
    group_size: int = 128
    high_count: int = 8
    middle_count: int = 2
    max_samples: int = 1
    max_tokens: int = 512
    seed: int = 42


def select_block_sensitivity_targets(
    diagnostics: dict[str, dict[str, Any]],
    *,
    high_count: int,
    middle_count: int,
) -> list[BlockSensitivityTarget]:
    """Select the highest-error leaves plus leaves nearest the global median."""

    if high_count <= 0 or middle_count <= 0:
        raise ValueError("high_count and middle_count must be positive")
    required = high_count + middle_count
    if len(diagnostics) < required:
        raise ValueError(
            f"need at least {required} eligible diagnostics, got {len(diagnostics)}"
        )
    ranked = sorted(
        diagnostics.items(),
        key=lambda item: (
            float(item[1]["awq_local_normalized_error"]),
            item[0],
        ),
        reverse=True,
    )
    high = ranked[:high_count]
    ascending = sorted(
        ranked,
        key=lambda item: (
            float(item[1]["awq_local_normalized_error"]),
            item[0],
        ),
    )
    high_names = {name for name, _ in high}
    center = (len(ascending) - 1) / 2
    middle_candidates = sorted(
        (
            (abs(index - center), index, item)
            for index, item in enumerate(ascending)
            if item[0] not in high_names
        ),
        key=lambda candidate: (candidate[0], candidate[1]),
    )
    middle = sorted(
        (candidate[2] for candidate in middle_candidates[:middle_count]),
        key=lambda item: (
            float(item[1]["awq_local_normalized_error"]),
            item[0],
        ),
    )

    def target(
        item: tuple[str, dict[str, Any]], selection: str
    ) -> BlockSensitivityTarget:
        name, payload = item
        return BlockSensitivityTarget(
            name=name,
            local_error=float(payload["awq_local_normalized_error"]),
            clip_ratio=float(payload["awq_clip_ratio"]),
            selection=selection,
        )

    return [
        *(target(item, "highest") for item in high),
        *(target(item, "middle") for item in middle),
    ]


def normalized_block_output_error(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    epsilon: float = 1e-12,
) -> tuple[float, float]:
    """Return the numerator and denominator used by normalized block error."""

    if reference.shape != candidate.shape:
        raise ValueError("reference and candidate block outputs must have equal shapes")
    reference_float = reference.float()
    candidate_float = candidate.float()
    numerator = float((reference_float - candidate_float).square().sum().item())
    denominator = float(reference_float.square().sum().clamp_min(epsilon).item())
    return numerator, denominator


def _hidden_from_block_output(output: object) -> torch.Tensor:
    value = output[0] if isinstance(output, (tuple, list)) else output
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"decoder block returned unsupported output: {type(value)}")
    return value


def _capture_block_outputs(
    model: torch.nn.Module,
    model_adapter: ModelAdapter,
    block_names: list[str],
    encoded_samples: list[dict[str, torch.Tensor]],
    *,
    device: str,
) -> dict[str, list[torch.Tensor]]:
    captured = {name: [] for name in block_names}
    handles: list[torch.utils.hooks.RemovableHandle] = []
    for block_name in block_names:
        block = model.get_submodule(block_name)

        def capture(
            module: torch.nn.Module,
            inputs: tuple[object, ...],
            output: object,
            name: str = block_name,
        ) -> None:
            del module, inputs
            captured[name].append(
                _hidden_from_block_output(output)
                .detach()
                .to(device="cpu", dtype=torch.bfloat16)
            )

        handles.append(block.register_forward_hook(capture))
    try:
        for encoded in encoded_samples:
            model_adapter.forward_hidden(
                model,
                **{key: value.to(device) for key, value in encoded.items()},
                use_cache=False,
            )
    finally:
        for handle in handles:
            handle.remove()
    expected = len(encoded_samples)
    for name, outputs in captured.items():
        if len(outputs) != expected:
            raise RuntimeError(
                f"captured {len(outputs)} outputs for {name}, expected {expected}"
            )
    return captured


def run_block_sensitivity(
    request: BlockSensitivityRequest,
    model_adapter: ModelAdapter,
) -> dict[str, object]:
    if not request.model:
        raise ValueError("model_id must be provided")
    if request.group_size <= 0:
        raise ValueError("group_size must be positive")
    if request.max_samples <= 0:
        raise ValueError("max_samples must be positive")
    if request.max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    with Path(request.awq_diagnostics).open("r", encoding="utf-8") as handle:
        raw_diagnostics = json.load(handle)
    eligible_diagnostics: dict[str, dict[str, Any]] = {}
    block_names_by_target: dict[str, str] = {}
    for name, payload in raw_diagnostics.items():
        try:
            block_name = model_adapter.block_name_for_module(name)
        except ValueError:
            continue
        eligible_diagnostics[name] = payload
        block_names_by_target[name] = block_name
    targets = select_block_sensitivity_targets(
        eligible_diagnostics,
        high_count=request.high_count,
        middle_count=request.middle_count,
    )

    device = cuda_device()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model = model_adapter.load_model(
        request.model,
        revision=request.revision,
        dtype=torch.bfloat16,
        device_map={"": device},
        attn_implementation=request.attn_implementation,
    )
    model.eval()
    load_seconds = time.perf_counter() - started
    tokenizer = AutoTokenizer.from_pretrained(request.model, revision=request.revision)
    rows = read_jsonl(request.calibration_jsonl)
    random.Random(request.seed).shuffle(rows)
    encoded_samples: list[dict[str, torch.Tensor]] = []
    for row in rows:
        text = model_adapter.render_prompt(
            tokenizer,
            row,
            thinking=False,
            reasoning_effort=None,
            force_chat_template=False,
        )
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=False,
            add_special_tokens=True,
        )
        if encoded.input_ids.shape[1] > request.max_tokens:
            continue
        encoded_samples.append(dict(encoded.items()))
        if len(encoded_samples) == request.max_samples:
            break
    if not encoded_samples:
        raise ValueError("no calibration sample fits the block-sensitivity token limit")

    unique_blocks = list(
        dict.fromkeys(block_names_by_target[target.name] for target in targets)
    )
    results: list[dict[str, object]] = []
    with torch.inference_mode():
        baseline = _capture_block_outputs(
            model,
            model_adapter,
            unique_blocks,
            encoded_samples,
            device=device,
        )
        for target in targets:
            source = model.get_submodule(target.name)
            if not isinstance(source, torch.nn.Linear):
                raise TypeError(
                    f"block sensitivity target is not Linear: {target.name}"
                )
            replacement = FrozenQuantLinear(
                source,
                bits=3,
                group_size=request.group_size,
                clip_ratio=target.clip_ratio,
            )
            parent, leaf = resolve_parent(model, target.name)
            setattr(parent, leaf, replacement)
            block_name = block_names_by_target[target.name]
            try:
                candidate = _capture_block_outputs(
                    model,
                    model_adapter,
                    [block_name],
                    encoded_samples,
                    device=device,
                )[block_name]
            finally:
                setattr(parent, leaf, source)
            numerator = denominator = 0.0
            for reference_output, candidate_output in zip(
                baseline[block_name], candidate, strict=True
            ):
                sample_numerator, sample_denominator = normalized_block_output_error(
                    reference_output,
                    candidate_output,
                )
                numerator += sample_numerator
                denominator += sample_denominator
            block_error = numerator / denominator
            result = {
                **asdict(target),
                "block_name": block_name,
                "block_error": block_error,
                "block_to_local_ratio": block_error / max(target.local_error, 1e-12),
            }
            results.append(result)
            print(
                f"BLOCK {target.name}: local={target.local_error:.8f} "
                f"block={block_error:.8f}",
                flush=True,
            )
            del replacement, candidate
            torch.cuda.empty_cache()
    torch.cuda.synchronize()
    payload: dict[str, object] = {
        "model": request.model,
        "revision": request.revision,
        "attn_implementation": request.attn_implementation,
        "group_size": request.group_size,
        "samples": len(encoded_samples),
        "sample_tokens": [
            int(encoded["input_ids"].numel()) for encoded in encoded_samples
        ],
        "load_seconds": load_seconds,
        "total_seconds": time.perf_counter() - started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "results": results,
    }
    output = Path(request.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return payload
