"""BF16 teacher-forcing phase for serial OPD."""

from __future__ import annotations

import gc
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import save_file

from ..model.base import ModelAdapter
from .artifacts import read_valid_rollouts, save_teacher_artifact
from .common import cuda_device
from .telemetry import cuda_memory_snapshot


@dataclass(slots=True)
class TeacherRequest:
    model: str = ""
    revision: str = "main"
    attn_implementation: str = "sdpa"
    rollouts: str | Path = "rollouts.jsonl"
    output: str | Path = "teacher_artifacts"
    max_tokens: int = 16_384


def run_teacher(
    request: TeacherRequest, model_adapter: ModelAdapter
) -> dict[str, int | float]:
    if not request.model:
        raise ValueError("teacher model_id must be provided")
    wall_started = time.perf_counter()
    device = cuda_device()
    records = read_valid_rollouts(request.rollouts, max_tokens=request.max_tokens)
    torch.cuda.reset_peak_memory_stats()
    setup_started = time.perf_counter()
    model = model_adapter.load_model(
        request.model,
        revision=request.revision,
        dtype=torch.bfloat16,
        device_map={"": device},
        attn_implementation=request.attn_implementation,
    )
    model.eval()
    torch.cuda.synchronize()
    setup_seconds = time.perf_counter() - setup_started
    setup_memory = cuda_memory_snapshot()
    torch.cuda.reset_peak_memory_stats()
    target = Path(request.output)
    target.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "weight": model.get_output_embeddings()
            .weight.detach()
            .cpu()
            .to(torch.bfloat16)
            .contiguous()
        },
        target / "teacher_lm_head.safetensors",
    )
    scoring_seconds = artifact_seconds = 0.0
    sequence_tokens = 0
    with torch.inference_mode():
        for record in records:
            input_ids = torch.tensor(
                record.input_ids, dtype=torch.long, device=device
            ).unsqueeze(0)
            torch.cuda.synchronize()
            scoring_started = time.perf_counter()
            hidden_states = model_adapter.forward_hidden(
                model,
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=False,
            )
            torch.cuda.synchronize()
            sample_scoring_seconds = time.perf_counter() - scoring_started
            scoring_seconds += sample_scoring_seconds
            sequence_tokens += len(record.input_ids)
            safe_id = hashlib.sha256(record.sample_id.encode("utf-8")).hexdigest()[:20]
            artifact_started = time.perf_counter()
            save_teacher_artifact(
                target / f"{safe_id}.safetensors",
                record=record,
                hidden_states=hidden_states,
            )
            sample_artifact_seconds = time.perf_counter() - artifact_started
            artifact_seconds += sample_artifact_seconds
            print(
                json.dumps(
                    {
                        "sample": record.sample_id,
                        "sequence_tokens": len(record.input_ids),
                        "scoring_seconds": sample_scoring_seconds,
                        "sequence_tokens_per_second": (
                            len(record.input_ids) / sample_scoring_seconds
                        ),
                        "artifact_seconds": sample_artifact_seconds,
                    }
                ),
                flush=True,
            )
            del hidden_states, input_ids
    phase_memory = cuda_memory_snapshot()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    artifact_bytes = sum(path.stat().st_size for path in target.glob("*.safetensors"))
    wall_seconds = time.perf_counter() - wall_started
    return {
        "teacher_artifacts": len(records),
        "sequence_tokens": sequence_tokens,
        "setup_seconds": setup_seconds,
        "scoring_seconds": scoring_seconds,
        "artifact_seconds": artifact_seconds,
        "wall_seconds": wall_seconds,
        "sequence_tokens_per_second": (
            sequence_tokens / scoring_seconds if scoring_seconds else 0.0
        ),
        "artifact_bytes": artifact_bytes,
        "setup_peak_allocated_gib": setup_memory.peak_allocated_gib,
        "scoring_peak_allocated_gib": phase_memory.peak_allocated_gib,
    }
