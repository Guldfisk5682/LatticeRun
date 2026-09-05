"""TensorBoard telemetry for OPD convergence, throughput, and memory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

_GIB = 1024**3


class ScalarWriter(Protocol):
    def add_scalar(
        self, tag: str, scalar_value: float, global_step: int | None = None
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class CudaMemorySnapshot:
    allocated_gib: float
    reserved_gib: float
    peak_allocated_gib: float
    peak_reserved_gib: float


def cuda_memory_snapshot() -> CudaMemorySnapshot:
    return CudaMemorySnapshot(
        allocated_gib=torch.cuda.memory_allocated() / _GIB,
        reserved_gib=torch.cuda.memory_reserved() / _GIB,
        peak_allocated_gib=torch.cuda.max_memory_allocated() / _GIB,
        peak_reserved_gib=torch.cuda.max_memory_reserved() / _GIB,
    )


class TrainingTelemetry:
    """Maintain cumulative counters and write consistently stepped scalars."""

    def __init__(self, writer: ScalarWriter, *, ema_decay: float = 0.95) -> None:
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1)")
        self.writer = writer
        self.ema_decay = ema_decay
        self.kl_ema: float | None = None
        self.sequence_tokens_seen = 0
        self.completion_tokens_seen = 0
        self._pending_kl_token_sum = 0.0
        self._pending_completion_tokens = 0

    def _write(self, metrics: dict[str, float], step: int) -> None:
        for tag, value in metrics.items():
            self.writer.add_scalar(tag, value, global_step=step)

    def state_dict(self) -> dict[str, float | int | None]:
        if self._pending_completion_tokens != 0:
            raise RuntimeError("telemetry can only checkpoint at optimizer boundaries")
        return {
            "kl_ema": self.kl_ema,
            "sequence_tokens_seen": self.sequence_tokens_seen,
            "completion_tokens_seen": self.completion_tokens_seen,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        if self._pending_completion_tokens != 0:
            raise RuntimeError("cannot restore telemetry with pending micro-steps")
        kl_ema = state.get("kl_ema")
        self.kl_ema = None if kl_ema is None else float(kl_ema)
        self.sequence_tokens_seen = int(state["sequence_tokens_seen"])
        self.completion_tokens_seen = int(state["completion_tokens_seen"])

    def log_micro_step(
        self,
        *,
        micro_step: int,
        epoch_progress: float,
        forward_kl: float,
        sequence_tokens: int,
        completion_tokens: int,
        elapsed_seconds: float,
        memory: CudaMemorySnapshot,
    ) -> dict[str, float]:
        if elapsed_seconds <= 0:
            raise ValueError("elapsed_seconds must be positive")
        if sequence_tokens <= 0 or completion_tokens <= 0:
            raise ValueError("token counts must be positive")
        self.kl_ema = (
            forward_kl
            if self.kl_ema is None
            else self.ema_decay * self.kl_ema + (1.0 - self.ema_decay) * forward_kl
        )
        self.sequence_tokens_seen += sequence_tokens
        self.completion_tokens_seen += completion_tokens
        self._pending_kl_token_sum += forward_kl * completion_tokens
        self._pending_completion_tokens += completion_tokens
        metrics = {
            "train/trajectory_forward_kl": forward_kl,
            "train/trajectory_forward_kl_ema": self.kl_ema,
            "progress/epoch": epoch_progress,
            "data/sequence_tokens": float(sequence_tokens),
            "data/completion_tokens": float(completion_tokens),
            "data/sequence_tokens_seen": float(self.sequence_tokens_seen),
            "data/completion_tokens_seen": float(self.completion_tokens_seen),
            "performance/micro_step_seconds": elapsed_seconds,
            "performance/sequence_tokens_per_second": sequence_tokens / elapsed_seconds,
            "performance/completion_tokens_per_second": completion_tokens
            / elapsed_seconds,
            "memory/allocated_gib": memory.allocated_gib,
            "memory/reserved_gib": memory.reserved_gib,
            "memory/peak_allocated_gib": memory.peak_allocated_gib,
            "memory/peak_reserved_gib": memory.peak_reserved_gib,
        }
        self._write(metrics, micro_step)
        return metrics

    def log_optimizer_step(
        self,
        *,
        optimizer_step: int,
        grad_norm: float,
        learning_rate: float,
        full_step_seconds: float,
        optimizer_update_seconds: float,
        completion_tokens: int,
        memory: CudaMemorySnapshot,
    ) -> dict[str, float]:
        if self._pending_completion_tokens == 0:
            raise RuntimeError("optimizer telemetry has no accumulated micro-steps")
        if completion_tokens != self._pending_completion_tokens:
            raise RuntimeError("optimizer-step completion-token count changed")
        if full_step_seconds <= 0 or optimizer_update_seconds <= 0:
            raise ValueError("optimizer-step timings must be positive")
        metrics = {
            "train/forward_kl_step_mean": self._pending_kl_token_sum
            / self._pending_completion_tokens,
            "train/grad_norm": grad_norm,
            "train/learning_rate": learning_rate,
            "data/optimizer_step_completion_tokens": float(completion_tokens),
            "performance/optimizer_step_seconds": full_step_seconds,
            "performance/optimizer_update_seconds": optimizer_update_seconds,
            "performance/optimizer_step_completion_tokens_per_second": (
                completion_tokens / full_step_seconds
            ),
            "memory/optimizer_step_peak_allocated_gib": memory.peak_allocated_gib,
            "memory/optimizer_step_peak_reserved_gib": memory.peak_reserved_gib,
        }
        self._write(metrics, optimizer_step)
        self._pending_kl_token_sum = 0.0
        self._pending_completion_tokens = 0
        return metrics
