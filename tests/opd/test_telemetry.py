import pytest
from latticerun.opd.telemetry import (
    CudaMemorySnapshot,
    TrainingTelemetry,
)


class _Writer:
    def __init__(self):
        self.scalars = {}

    def add_scalar(self, tag, scalar_value, global_step=None):
        self.scalars[(tag, global_step)] = scalar_value


def _memory():
    return CudaMemorySnapshot(
        allocated_gib=5, reserved_gib=6, peak_allocated_gib=7, peak_reserved_gib=8
    )


def test_training_telemetry_tracks_loss_tokens_throughput_and_memory():
    writer = _Writer()
    telemetry = TrainingTelemetry(writer, ema_decay=0.5)
    first = telemetry.log_micro_step(
        micro_step=1,
        epoch_progress=0.25,
        forward_kl=2.0,
        sequence_tokens=100,
        completion_tokens=40,
        elapsed_seconds=2.0,
        memory=_memory(),
    )
    second = telemetry.log_micro_step(
        micro_step=2,
        epoch_progress=0.5,
        forward_kl=0.0,
        sequence_tokens=80,
        completion_tokens=20,
        elapsed_seconds=1.0,
        memory=_memory(),
    )
    optimizer = telemetry.log_optimizer_step(
        optimizer_step=1,
        grad_norm=3.0,
        learning_rate=2e-4,
        full_step_seconds=2.0,
        optimizer_update_seconds=0.25,
        completion_tokens=60,
        memory=_memory(),
    )

    assert first["performance/sequence_tokens_per_second"] == 50
    assert second["train/trajectory_forward_kl_ema"] == 1
    assert second["data/sequence_tokens_seen"] == 180
    assert second["data/completion_tokens_seen"] == 60
    assert second["memory/peak_reserved_gib"] == 8
    assert optimizer["train/forward_kl_step_mean"] == pytest.approx(4 / 3)
    assert optimizer["performance/optimizer_step_seconds"] == 2.0
    assert optimizer["performance/optimizer_update_seconds"] == 0.25
    assert optimizer["performance/optimizer_step_completion_tokens_per_second"] == 30
    assert optimizer["memory/optimizer_step_peak_allocated_gib"] == 7
    assert writer.scalars[("train/learning_rate", 1)] == 2e-4


def test_training_telemetry_rejects_invalid_measurements():
    telemetry = TrainingTelemetry(_Writer())
    with pytest.raises(ValueError, match="token counts"):
        telemetry.log_micro_step(
            micro_step=1,
            epoch_progress=1,
            forward_kl=1,
            sequence_tokens=1,
            completion_tokens=0,
            elapsed_seconds=1,
            memory=_memory(),
        )
