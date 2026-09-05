"""Atomic single-retention checkpoints for serial OPD training."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from ..adapters import load_adapter, save_adapter
from ..config import RecoveryConfig

_CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")


@dataclass(slots=True)
class TrainingProgress:
    global_step: int = 0
    micro_step: int = 0
    next_epoch: int = 0
    next_group_start: int = 0
    next_shard_index: int = 0
    optimizer_step_seconds_total: float = 0.0
    training_peak_allocated_gib: float = 0.0
    training_peak_reserved_gib: float = 0.0


def checkpoint_interval_steps(total_steps: int, fraction: float) -> int:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("checkpoint interval fraction must be in (0, 1]")
    return max(1, math.ceil(total_steps * fraction))


def artifact_fingerprint(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(path.stat().st_size).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _checkpoint_directories(output: Path) -> list[tuple[int, Path]]:
    checkpoints: list[tuple[int, Path]] = []
    if not output.exists():
        return checkpoints
    for path in output.iterdir():
        match = _CHECKPOINT_RE.fullmatch(path.name)
        if path.is_dir() and match:
            checkpoints.append((int(match.group(1)), path))
    return sorted(checkpoints)


def _prune_checkpoints(output: Path, save_total_limit: int) -> None:
    if save_total_limit <= 0:
        raise ValueError("save_total_limit must be positive")
    checkpoints = _checkpoint_directories(output)
    for _, path in checkpoints[:-save_total_limit]:
        shutil.rmtree(path)


def save_training_checkpoint(
    *,
    model: torch.nn.Module,
    recovery: RecoveryConfig,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    output: str | Path,
    progress: TrainingProgress,
    telemetry_state: dict[str, float | int | None],
    contract: dict[str, object],
    save_total_limit: int,
) -> tuple[Path, float]:
    """Save at an optimizer boundary, publish atomically, then prune older saves."""

    if progress.global_step <= 0:
        raise ValueError("checkpoint global_step must be positive")
    target_root = Path(output)
    target_root.mkdir(parents=True, exist_ok=True)
    destination = target_root / f"checkpoint-{progress.global_step}"
    if destination.exists():
        raise FileExistsError(f"checkpoint already exists: {destination}")
    started = time.perf_counter()
    temporary = Path(
        tempfile.mkdtemp(prefix=f".checkpoint-{progress.global_step}-", dir=target_root)
    )
    try:
        save_adapter(model, temporary / "adapter", recovery)
        torch.save(optimizer.state_dict(), temporary / "optimizer.pt")
        torch.save(scheduler.state_dict(), temporary / "scheduler.pt")
        torch.save(
            {
                "python": random.getstate(),
                "torch": torch.random.get_rng_state(),
                "cuda": (
                    torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
                ),
            },
            temporary / "rng.pt",
        )
        state = {
            "progress": asdict(progress),
            "telemetry": telemetry_state,
            "contract": contract,
        }
        with (temporary / "trainer_state.json").open("w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2, sort_keys=True)
            handle.write("\n")
        temporary.rename(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    _prune_checkpoints(target_root, save_total_limit)
    return destination, time.perf_counter() - started


def load_training_checkpoint(
    *,
    checkpoint: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    expected_contract: dict[str, object],
    device: torch.device,
) -> tuple[TrainingProgress, dict[str, object]]:
    source = Path(checkpoint)
    with (source / "trainer_state.json").open("r", encoding="utf-8") as handle:
        state = json.load(handle)
    if state["contract"] != expected_contract:
        raise ValueError("resume checkpoint training contract does not match this run")
    load_adapter(model, source / "adapter")
    optimizer.load_state_dict(
        torch.load(source / "optimizer.pt", map_location=device, weights_only=True)
    )
    scheduler.load_state_dict(
        torch.load(source / "scheduler.pt", map_location="cpu", weights_only=True)
    )
    rng = torch.load(source / "rng.pt", map_location="cpu", weights_only=False)
    random.setstate(rng["python"])
    torch.random.set_rng_state(rng["torch"])
    if rng["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["cuda"])
    return TrainingProgress(**state["progress"]), state["telemetry"]
