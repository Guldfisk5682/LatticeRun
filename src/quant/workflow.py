"""AWQ calibration and reference checkpoint export workflow."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoTokenizer

from ..config import QuantConfig
from ..model.base import ModelAdapter
from ..opd.common import cuda_device, read_jsonl
from .calibration import ActivationCollector
from .clipping import (
    choose_awq_clip_ratio,
    choose_weight_mse_clip_ratio,
    normalized_output_error,
)
from .export import export_quantized_checkpoint


@dataclass(slots=True)
class QuantizeRequest:
    model: str = ""
    revision: str = "main"
    attn_implementation: str = "sdpa"
    calibration_jsonl: str | Path = "calibration.jsonl"
    output: str | Path = "quantized"
    group_size: int = 128
    calibration_tokens: int = 512
    max_samples: int = 32
    max_tokens: int = 16_384
    seed: int = 42
    enable_deltanet_z: bool = False


def run_quantization(request: QuantizeRequest, model_adapter: ModelAdapter) -> None:
    if not request.model:
        raise ValueError("model_id must be provided")
    device = cuda_device()
    quant_config = QuantConfig(group_size=request.group_size)
    model = model_adapter.load_model(
        request.model,
        revision=request.revision,
        dtype=torch.bfloat16,
        device_map={"": device},
        attn_implementation=request.attn_implementation,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(request.model, revision=request.revision)
    decisions = model_adapter.inspect_named_modules(
        model.named_modules(),
        enable_deltanet_z=request.enable_deltanet_z,
        strict=True,
    )
    module_map = dict(model.named_modules())
    capture_decisions = [
        item
        for item in decisions
        if item.bits is not None and isinstance(module_map[item.name], torch.nn.Linear)
    ]
    rows = read_jsonl(request.calibration_jsonl)
    random.Random(request.seed).shuffle(rows)
    rows = rows[: request.max_samples]
    if not rows:
        raise ValueError("calibration JSONL is empty")
    with (
        ActivationCollector(
            model,
            capture_decisions,
            max_tokens_per_module=request.calibration_tokens,
        ) as collector,
        torch.inference_mode(),
    ):
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
            model(
                **{key: value.to(device) for key, value in encoded.items()},
                use_cache=False,
                return_dict=True,
            )
    ratios: dict[str, float] = {}
    diagnostics: dict[str, dict[str, float]] = {}
    for decision in capture_decisions:
        activation = collector.merged(decision.name)
        weight = module_map[decision.name].weight
        bits = decision.bits or quant_config.bits
        ratio, errors = choose_awq_clip_ratio(
            weight,
            activation,
            bits=bits,
            group_size=quant_config.group_size,
            ratios=quant_config.awq.clip_ratios,
            token_batch_size=quant_config.awq.token_batch_size,
        )
        ratios[decision.name] = ratio
        weight_ratio, weight_errors = choose_weight_mse_clip_ratio(
            weight,
            bits=bits,
            group_size=quant_config.group_size,
            ratios=quant_config.awq.clip_ratios,
        )
        diagnostics[decision.name] = {
            "awq_clip_ratio": ratio,
            "awq_local_normalized_error": normalized_output_error(
                weight,
                activation,
                bits=bits,
                group_size=quant_config.group_size,
                clip_ratio=ratio,
            ),
            "weight_mse_clip_ratio": weight_ratio,
            "weight_mse_local_normalized_error": normalized_output_error(
                weight,
                activation,
                bits=bits,
                group_size=quant_config.group_size,
                clip_ratio=weight_ratio,
            ),
            **{f"awq_mse_{key}": value for key, value in errors.items()},
            **{f"weight_mse_{key}": value for key, value in weight_errors.items()},
        }
        print(f"AWQ {decision.name}: {ratio:.2f}", flush=True)
    output = Path(request.output)
    output.mkdir(parents=True, exist_ok=True)
    for filename, payload in (
        ("awq_clip_ratios.json", ratios),
        ("awq_diagnostics.json", diagnostics),
    ):
        with (output / filename).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
    export_quantized_checkpoint(
        model,
        decisions,
        output,
        group_size=quant_config.group_size,
        clip_ratios=ratios,
    )
