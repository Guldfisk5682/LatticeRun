"""Serial student-training phase for exact full-vocabulary OPD."""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from ..adapters import save_adapter, trainable_adapter_parameters
from ..model.base import ModelAdapter
from .artifacts import load_rollout_shard_plan
from .checkpoint import (
    TrainingProgress,
    artifact_fingerprint,
    checkpoint_interval_steps,
    load_training_checkpoint,
    save_training_checkpoint,
)
from .common import cuda_device
from .loss import backward_chunked_forward_kl
from .student import StudentRequest, load_student
from .telemetry import TrainingTelemetry, cuda_memory_snapshot


@dataclass(slots=True)
class TrainerRequest:
    student: StudentRequest = field(default_factory=StudentRequest)
    teacher_artifacts: str | Path = "teacher_artifacts"
    output: str | Path = "adapter"
    token_chunk_size: int = 256
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    gradient_accumulation_steps: int = 8
    epochs: int = 1
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True
    seed: int = 42
    logging_dir: str | Path | None = None
    logging_flush_secs: int = 10
    checkpoint_interval_fraction: float = 1 / 3
    save_total_limit: int = 1
    resume_from_checkpoint: str | Path | None = None
    training_plan: str | Path | None = None
    shard_index: int | None = None


def _artifact_completion_tokens(path: Path) -> int:
    with safe_open(path, framework="pt", device="cpu") as handle:
        mask = handle.get_tensor("completion_mask")
    if mask.ndim != 1 or mask.numel() < 2:
        raise ValueError(f"invalid completion mask in {path}")
    count = int(mask[1:].sum().item())
    if count <= 0:
        raise ValueError(f"teacher artifact has no supervised tokens: {path}")
    return count


def _warmup_steps(total_steps: int, warmup_ratio: float) -> int:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if not 0.0 <= warmup_ratio < 1.0:
        raise ValueError("warmup_ratio must be in [0, 1)")
    return math.ceil(total_steps * warmup_ratio)


def _artifact_sample_ids(paths: list[Path]) -> set[str]:
    sample_ids: set[str] = set()
    for path in paths:
        with safe_open(path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
        sample_id = metadata.get("sample_id")
        if not sample_id or sample_id in sample_ids:
            raise ValueError(f"missing or duplicate sample_id metadata: {path}")
        sample_ids.add(sample_id)
    return sample_ids


def run_training(
    request: TrainerRequest, model_adapter: ModelAdapter
) -> dict[str, int | float | str]:
    from torch.utils.tensorboard import SummaryWriter

    if request.gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    if request.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if request.weight_decay != 0.0:
        raise ValueError("the initial OPD recipe fixes weight_decay at zero")
    if not 0.0 <= request.warmup_ratio < 1.0:
        raise ValueError("warmup_ratio must be in [0, 1)")
    if request.lr_scheduler_type != "cosine":
        raise ValueError("the initial OPD recipe uses cosine learning-rate decay")
    if request.epochs <= 0:
        raise ValueError("epochs must be positive")
    if request.max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive")
    if request.logging_flush_secs <= 0:
        raise ValueError("logging_flush_secs must be positive")
    if not 0.0 < request.checkpoint_interval_fraction <= 1.0:
        raise ValueError("checkpoint_interval_fraction must be in (0, 1]")
    if request.save_total_limit <= 0:
        raise ValueError("save_total_limit must be positive")
    if (request.training_plan is None) != (request.shard_index is None):
        raise ValueError("training_plan and shard_index must be provided together")
    random.seed(request.seed)
    torch.manual_seed(request.seed)
    torch.cuda.manual_seed_all(request.seed)
    device = cuda_device()
    model = load_student(request.student, model_adapter, for_training=True)
    model.train()
    if request.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    model.config.use_cache = False
    parameters = trainable_adapter_parameters(model)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=request.learning_rate,
        weight_decay=request.weight_decay,
        fused=True,
    )
    teacher_dir = Path(request.teacher_artifacts)
    teacher_head = load_file(teacher_dir / "teacher_lm_head.safetensors")["weight"].to(
        device=device, dtype=torch.bfloat16
    )
    base_artifacts = sorted(
        path
        for path in teacher_dir.glob("*.safetensors")
        if path.name != "teacher_lm_head.safetensors"
    )
    if not base_artifacts:
        raise ValueError("teacher artifact directory is empty")
    shard_plan: dict[str, object] | None = None
    plan_fingerprint: str | None = None
    shard_count = 1
    total_training_samples = len(base_artifacts)
    if request.training_plan is not None:
        shard_plan, plan_fingerprint = load_rollout_shard_plan(request.training_plan)
        shard_count = len(shard_plan["shards"])
        assert request.shard_index is not None
        if not 0 <= request.shard_index < shard_count:
            raise ValueError("shard_index is outside the training plan")
        shard = shard_plan["shards"][request.shard_index]
        expected_ids = set(shard["sample_ids"])
        observed_ids = _artifact_sample_ids(base_artifacts)
        if observed_ids != expected_ids:
            raise ValueError(
                "teacher artifacts do not match the requested rollout shard"
            )
        if int(shard_plan["gradient_accumulation_steps"]) != (
            request.gradient_accumulation_steps
        ):
            raise ValueError("training plan gradient accumulation does not match")
        if request.epochs != 1:
            raise ValueError("sharded OPD currently supports exactly one epoch")
        total_training_samples = int(shard_plan["total_samples"])
    completion_tokens_by_path = {
        path: _artifact_completion_tokens(path) for path in base_artifacts
    }
    estimated_updates = (
        (total_training_samples + request.gradient_accumulation_steps - 1)
        // request.gradient_accumulation_steps
        * request.epochs
    )
    warmup_steps = _warmup_steps(estimated_updates, request.warmup_ratio)
    checkpoint_steps = checkpoint_interval_steps(
        estimated_updates, request.checkpoint_interval_fraction
    )
    contract: dict[str, object] = {
        "artifact_fingerprint": (
            plan_fingerprint
            if plan_fingerprint is not None
            else artifact_fingerprint(base_artifacts)
        ),
        "training_samples": total_training_samples,
        "estimated_optimizer_steps": estimated_updates,
        "gradient_accumulation_steps": request.gradient_accumulation_steps,
        "epochs": request.epochs,
        "seed": request.seed,
        "learning_rate": request.learning_rate,
        "weight_decay": request.weight_decay,
        "warmup_ratio": request.warmup_ratio,
        "lr_scheduler_type": request.lr_scheduler_type,
        "max_grad_norm": request.max_grad_norm,
        "token_chunk_size": request.token_chunk_size,
        "adapter_type": request.student.recovery.adapter_type,
        "adapter_mode": request.student.recovery.mode,
        "adapter_rank": request.student.recovery.rank,
        "adapter_alpha": request.student.recovery.alpha,
    }
    from transformers import get_cosine_schedule_with_warmup

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=estimated_updates,
    )
    progress = TrainingProgress()
    restored_telemetry_state: dict[str, object] | None = None
    if request.resume_from_checkpoint is not None:
        progress, restored_telemetry_state = load_training_checkpoint(
            checkpoint=request.resume_from_checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_contract=contract,
            device=device,
        )
    if request.shard_index is not None and progress.next_shard_index != (
        request.shard_index
    ):
        raise ValueError(
            "checkpoint expects shard "
            f"{progress.next_shard_index}, received {request.shard_index}"
        )
    logging_dir = Path(request.logging_dir or Path(request.output) / "runs")
    writer = SummaryWriter(
        log_dir=str(logging_dir),
        max_queue=10,
        flush_secs=request.logging_flush_secs,
    )
    telemetry = TrainingTelemetry(writer)
    if restored_telemetry_state is not None:
        telemetry.load_state_dict(restored_telemetry_state)
    writer.add_text(
        "run/config",
        "```json\n"
        + json.dumps(
            {
                "model": request.student.model,
                "revision": request.student.revision,
                "optimizer": "AdamW",
                "adapter_type": request.student.recovery.adapter_type,
                "adapter_mode": request.student.recovery.mode,
                "adapter_rank": request.student.recovery.rank,
                "adapter_alpha": request.student.recovery.alpha,
                "token_chunk_size": request.token_chunk_size,
                "learning_rate": request.learning_rate,
                "weight_decay": request.weight_decay,
                "warmup_ratio": request.warmup_ratio,
                "warmup_steps": warmup_steps,
                "lr_scheduler_type": request.lr_scheduler_type,
                "gradient_accumulation_steps": request.gradient_accumulation_steps,
                "max_grad_norm": request.max_grad_norm,
                "loss_reduction": "global_valid_completion_token_mean",
                "epochs": request.epochs,
                "training_samples": total_training_samples,
                "current_shard_index": request.shard_index,
                "shard_count": shard_count,
                "estimated_optimizer_steps": estimated_updates,
                "checkpoint_interval_fraction": (request.checkpoint_interval_fraction),
                "checkpoint_interval_steps": checkpoint_steps,
                "save_total_limit": request.save_total_limit,
                "resume_from_checkpoint": (
                    str(request.resume_from_checkpoint)
                    if request.resume_from_checkpoint is not None
                    else None
                ),
            },
            indent=2,
        )
        + "\n```",
        0,
    )
    writer.add_scalar("run/estimated_optimizer_steps", estimated_updates, 0)
    writer.add_scalar("run/warmup_steps", warmup_steps, 0)
    writer.add_scalar("run/training_samples", total_training_samples, 0)
    writer.add_scalar("run/checkpoint_interval_steps", checkpoint_steps, 0)
    writer.add_scalar("run/resumed_global_step", progress.global_step, 0)
    setup_memory = cuda_memory_snapshot()
    writer.add_scalar("memory/setup_allocated_gib", setup_memory.allocated_gib, 0)
    writer.add_scalar("memory/setup_reserved_gib", setup_memory.reserved_gib, 0)
    writer.add_scalar(
        "memory/setup_peak_allocated_gib", setup_memory.peak_allocated_gib, 0
    )
    writer.add_scalar(
        "memory/setup_peak_reserved_gib", setup_memory.peak_reserved_gib, 0
    )
    optimizer.zero_grad(set_to_none=True)
    # Preserve setup/load peaks above, then make per-step peak fields describe
    # the actual training phase rather than transient checkpoint construction.
    torch.cuda.reset_peak_memory_stats()
    global_step = progress.global_step
    micro_step = progress.micro_step
    optimizer_step_seconds_total = progress.optimizer_step_seconds_total
    training_peak_allocated_gib = progress.training_peak_allocated_gib
    training_peak_reserved_gib = progress.training_peak_reserved_gib
    checkpoint_seconds_total = 0.0
    last_checkpoint: Path | None = None
    try:
        for epoch in range(progress.next_epoch, request.epochs):
            artifacts = list(base_artifacts)
            random.Random(request.seed + epoch).shuffle(artifacts)
            first_group_start = (
                progress.next_group_start if epoch == progress.next_epoch else 0
            )
            for group_start in range(
                first_group_start,
                len(artifacts),
                request.gradient_accumulation_steps,
            ):
                group = artifacts[
                    group_start : group_start + request.gradient_accumulation_steps
                ]
                group_completion_tokens = sum(
                    completion_tokens_by_path[path] for path in group
                )
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                full_optimizer_step_started = time.perf_counter()
                for group_index, path in enumerate(group):
                    sample_index = group_start + group_index + 1
                    torch.cuda.synchronize()
                    micro_step_started = time.perf_counter()
                    artifact = load_file(path)
                    input_ids = (
                        artifact["input_ids"]
                        .to(device=device, dtype=torch.long)
                        .unsqueeze(0)
                    )
                    completion_mask = (
                        artifact["completion_mask"].to(device=device).unsqueeze(0)
                    )
                    teacher_hidden = (
                        artifact["teacher_hidden"]
                        .to(device=device, dtype=torch.bfloat16)
                        .unsqueeze(0)
                    )
                    completion_tokens = int(completion_mask[:, 1:].sum().item())
                    if completion_tokens != completion_tokens_by_path[path]:
                        raise RuntimeError(f"completion-token count changed: {path}")
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        student_hidden = model_adapter.forward_hidden(
                            model,
                            input_ids=input_ids,
                            attention_mask=torch.ones_like(input_ids),
                            use_cache=False,
                        )
                    loss = backward_chunked_forward_kl(
                        student_hidden,
                        teacher_hidden,
                        model.get_output_embeddings().weight,
                        teacher_head,
                        completion_mask,
                        token_chunk_size=request.token_chunk_size,
                        gradient_denominator=group_completion_tokens,
                    )
                    micro_step += 1
                    should_step = group_index == len(group) - 1
                    optimizer_step_seconds = 0.0
                    full_optimizer_step_seconds = 0.0
                    optimizer_step_memory = None
                    grad_norm_value: float | None = None
                    if should_step:
                        torch.cuda.synchronize()
                        optimizer_step_started = time.perf_counter()
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            parameters, request.max_grad_norm
                        )
                        step_learning_rate = float(optimizer.param_groups[0]["lr"])
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                        torch.cuda.synchronize()
                        optimizer_step_seconds = (
                            time.perf_counter() - optimizer_step_started
                        )
                        full_optimizer_step_seconds = (
                            time.perf_counter() - full_optimizer_step_started
                        )
                        optimizer_step_memory = cuda_memory_snapshot()
                        optimizer_step_seconds_total += full_optimizer_step_seconds
                        training_peak_allocated_gib = max(
                            training_peak_allocated_gib,
                            optimizer_step_memory.peak_allocated_gib,
                        )
                        training_peak_reserved_gib = max(
                            training_peak_reserved_gib,
                            optimizer_step_memory.peak_reserved_gib,
                        )
                        grad_norm_value = float(grad_norm)
                        global_step += 1
                    torch.cuda.synchronize()
                    micro_step_seconds = time.perf_counter() - micro_step_started
                    sequence_tokens = int(input_ids.numel())
                    micro_metrics = telemetry.log_micro_step(
                        micro_step=micro_step,
                        epoch_progress=epoch + sample_index / len(base_artifacts),
                        forward_kl=loss,
                        sequence_tokens=sequence_tokens,
                        completion_tokens=completion_tokens,
                        elapsed_seconds=micro_step_seconds,
                        memory=cuda_memory_snapshot(),
                    )
                    if should_step:
                        if grad_norm_value is None or optimizer_step_memory is None:
                            raise RuntimeError(
                                "optimizer step did not produce complete telemetry"
                            )
                        telemetry.log_optimizer_step(
                            optimizer_step=global_step,
                            grad_norm=grad_norm_value,
                            # This is the LR used by the optimizer update just
                            # completed, not the next scheduler value.
                            learning_rate=step_learning_rate,
                            full_step_seconds=full_optimizer_step_seconds,
                            optimizer_update_seconds=optimizer_step_seconds,
                            completion_tokens=group_completion_tokens,
                            memory=optimizer_step_memory,
                        )
                    print(
                        json.dumps(
                            {
                                "epoch": epoch,
                                "sample": path.stem,
                                "trajectory_kl": loss,
                                "step": global_step,
                                "estimated_updates": estimated_updates,
                                "sequence_tokens": sequence_tokens,
                                "completion_tokens": completion_tokens,
                                "accumulation_group_completion_tokens": (
                                    group_completion_tokens
                                ),
                                "micro_step_seconds": micro_step_seconds,
                                "sequence_tokens_per_second": micro_metrics[
                                    "performance/sequence_tokens_per_second"
                                ],
                                "allocated_gib": micro_metrics["memory/allocated_gib"],
                                "peak_allocated_gib": micro_metrics[
                                    "memory/peak_allocated_gib"
                                ],
                                "optimizer_step_seconds": (
                                    full_optimizer_step_seconds if should_step else None
                                ),
                                "optimizer_step_completion_tokens_per_second": (
                                    group_completion_tokens
                                    / full_optimizer_step_seconds
                                    if should_step
                                    else None
                                ),
                                "optimizer_update_seconds": (
                                    optimizer_step_seconds if should_step else None
                                ),
                                "optimizer_step_peak_allocated_gib": (
                                    optimizer_step_memory.peak_allocated_gib
                                    if optimizer_step_memory is not None
                                    else None
                                ),
                                "optimizer_step_peak_reserved_gib": (
                                    optimizer_step_memory.peak_reserved_gib
                                    if optimizer_step_memory is not None
                                    else None
                                ),
                            }
                        ),
                        flush=True,
                    )
                    del (
                        artifact,
                        input_ids,
                        completion_mask,
                        teacher_hidden,
                        student_hidden,
                    )
                    if should_step:
                        next_epoch = epoch
                        next_group_start = group_start + len(group)
                        next_shard_index = progress.next_shard_index
                        shard_completed = next_group_start >= len(artifacts)
                        if shard_completed:
                            next_epoch = 0 if shard_plan is not None else next_epoch + 1
                            next_group_start = 0
                            if shard_plan is not None:
                                next_shard_index += 1
                        progress = TrainingProgress(
                            global_step=global_step,
                            micro_step=micro_step,
                            next_epoch=next_epoch,
                            next_group_start=next_group_start,
                            next_shard_index=next_shard_index,
                            optimizer_step_seconds_total=(optimizer_step_seconds_total),
                            training_peak_allocated_gib=(training_peak_allocated_gib),
                            training_peak_reserved_gib=training_peak_reserved_gib,
                        )
                        checkpoint_due = (
                            global_step % checkpoint_steps == 0
                            or global_step == estimated_updates
                            or (shard_plan is not None and shard_completed)
                        )
                        if checkpoint_due:
                            writer.flush()
                            last_checkpoint, checkpoint_seconds = (
                                save_training_checkpoint(
                                    model=model,
                                    recovery=request.student.recovery,
                                    optimizer=optimizer,
                                    scheduler=scheduler,
                                    output=request.output,
                                    progress=progress,
                                    telemetry_state=telemetry.state_dict(),
                                    contract=contract,
                                    save_total_limit=request.save_total_limit,
                                )
                            )
                            checkpoint_seconds_total += checkpoint_seconds
                            writer.add_scalar(
                                "performance/checkpoint_seconds",
                                checkpoint_seconds,
                                global_step,
                            )
                            print(
                                json.dumps(
                                    {
                                        "checkpoint": str(last_checkpoint),
                                        "step": global_step,
                                        "checkpoint_seconds": checkpoint_seconds,
                                        "save_total_limit": (request.save_total_limit),
                                    }
                                ),
                                flush=True,
                            )
    finally:
        writer.flush()
        writer.close()
    save_adapter(model, request.output, request.student.recovery)
    return {
        "optimizer_steps": global_step,
        "micro_steps": micro_step,
        "sequence_tokens": telemetry.sequence_tokens_seen,
        "completion_tokens": telemetry.completion_tokens_seen,
        "optimizer_step_seconds": optimizer_step_seconds_total,
        "completion_tokens_per_optimizer_step_second": (
            telemetry.completion_tokens_seen / optimizer_step_seconds_total
            if optimizer_step_seconds_total
            else 0.0
        ),
        "training_peak_allocated_gib": training_peak_allocated_gib,
        "training_peak_reserved_gib": training_peak_reserved_gib,
        "checkpoint_interval_steps": checkpoint_steps,
        "checkpoint_seconds": checkpoint_seconds_total,
        "last_checkpoint": str(last_checkpoint) if last_checkpoint else "",
    }
