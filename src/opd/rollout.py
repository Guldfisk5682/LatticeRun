"""On-policy student rollout generation."""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from transformers import AutoTokenizer

from ..model.base import ModelAdapter, ReasoningEffort
from .artifacts import RolloutRecord
from .common import cuda_device, read_jsonl
from .student import StudentRequest, load_student
from .telemetry import cuda_memory_snapshot


@dataclass(slots=True)
class RolloutRequest:
    student: StudentRequest = field(default_factory=StudentRequest)
    prompts: str | Path = "prompts.jsonl"
    output: str | Path = "rollouts.jsonl"
    max_tokens: int = 16_384
    max_new_tokens: int = 4096
    thinking: bool = True
    reasoning_effort: ReasoningEffort = "medium"
    do_sample: bool = True
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 20
    seed: int = 42
    backend: str = "transformers"
    vllm_gpu_memory_utilization: float = 0.90
    vllm_max_num_seqs: int = 10
    vllm_enforce_eager: bool = False
    vllm_attention_backend: str = "FLASH_ATTN"
    vllm_flash_attn_version: int = 2
    vllm_use_flashinfer_sampler: bool = False


def run_rollout(
    request: RolloutRequest, model_adapter: ModelAdapter
) -> dict[str, int | float]:
    wall_started = time.perf_counter()
    random.seed(request.seed)
    torch.manual_seed(request.seed)
    torch.cuda.reset_peak_memory_stats()
    setup_started = time.perf_counter()
    model = load_student(request.student, model_adapter, for_training=False)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        request.student.model, revision=request.student.revision
    )
    torch.cuda.synchronize()
    setup_seconds = time.perf_counter() - setup_started
    setup_memory = cuda_memory_snapshot()
    torch.cuda.reset_peak_memory_stats()
    eos_ids = tokenizer.eos_token_id
    eos_set = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])
    rows = read_jsonl(request.prompts)
    output = Path(request.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    kept = dropped = 0
    prompt_tokens_seen = 0
    generated_tokens = 0
    generation_seconds = 0.0
    device = cuda_device()
    with output.open("w", encoding="utf-8") as handle, torch.inference_mode():
        for index, row in enumerate(rows):
            prompt = model_adapter.render_prompt(
                tokenizer,
                row,
                thinking=request.thinking,
                reasoning_effort=request.reasoning_effort,
                force_chat_template=True,
            )
            encoded = tokenizer(prompt, return_tensors="pt", truncation=False)
            prompt_tokens = encoded.input_ids.shape[1]
            if prompt_tokens >= request.max_tokens:
                dropped += 1
                continue
            remaining = min(request.max_new_tokens, request.max_tokens - prompt_tokens)
            torch.cuda.synchronize()
            generation_started = time.perf_counter()
            output_ids = model.generate(
                input_ids=encoded.input_ids.to(device),
                attention_mask=encoded.attention_mask.to(device),
                max_new_tokens=remaining,
                do_sample=request.do_sample,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                use_cache=True,
            )[0].tolist()
            torch.cuda.synchronize()
            sample_seconds = time.perf_counter() - generation_started
            sample_generated_tokens = len(output_ids) - prompt_tokens
            prompt_tokens_seen += prompt_tokens
            generated_tokens += sample_generated_tokens
            generation_seconds += sample_seconds
            eos_reached = bool(output_ids and output_ids[-1] in eos_set)
            record = RolloutRecord(
                sample_id=str(row.get("id", index)),
                input_ids=output_ids,
                completion_start=prompt_tokens,
                eos_reached=eos_reached,
                truncated=not eos_reached or len(output_ids) > request.max_tokens,
            )
            if record.truncated:
                dropped += 1
                print(
                    json.dumps(
                        {
                            "sample": record.sample_id,
                            "kept": False,
                            "prompt_tokens": prompt_tokens,
                            "generated_tokens": sample_generated_tokens,
                            "generation_seconds": sample_seconds,
                            "generated_tokens_per_second_including_prefill": (
                                sample_generated_tokens / sample_seconds
                            ),
                        }
                    ),
                    flush=True,
                )
                continue
            record.validate(max_tokens=request.max_tokens)
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
            kept += 1
            print(
                json.dumps(
                    {
                        "sample": record.sample_id,
                        "kept": True,
                        "prompt_tokens": prompt_tokens,
                        "generated_tokens": sample_generated_tokens,
                        "generation_seconds": sample_seconds,
                        "generated_tokens_per_second_including_prefill": (
                            sample_generated_tokens / sample_seconds
                        ),
                    }
                ),
                flush=True,
            )
    phase_memory = cuda_memory_snapshot()
    wall_seconds = time.perf_counter() - wall_started
    return {
        "kept": kept,
        "discarded_truncated_or_oversize": dropped,
        "prompt_tokens": int(prompt_tokens_seen),
        "generated_tokens": int(generated_tokens),
        "setup_seconds": setup_seconds,
        "generation_seconds": generation_seconds,
        "wall_seconds": wall_seconds,
        "generated_tokens_per_second_including_prefill": (
            generated_tokens / generation_seconds if generation_seconds else 0.0
        ),
        "setup_peak_allocated_gib": setup_memory.peak_allocated_gib,
        "generation_peak_allocated_gib": phase_memory.peak_allocated_gib,
    }
