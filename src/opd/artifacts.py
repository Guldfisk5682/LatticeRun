"""Validated rollout records and compact teacher artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(slots=True)
class RolloutRecord:
    sample_id: str
    input_ids: list[int]
    completion_start: int
    eos_reached: bool
    truncated: bool

    def validate(self, max_tokens: int = 16_384) -> None:
        if self.truncated:
            raise ValueError(f"truncated rollout must be discarded: {self.sample_id}")
        if not self.eos_reached:
            raise ValueError(f"rollout did not reach EOS: {self.sample_id}")
        if len(self.input_ids) > max_tokens:
            raise ValueError(f"rollout exceeds max_tokens: {self.sample_id}")
        if not 1 <= self.completion_start < len(self.input_ids):
            raise ValueError(f"invalid completion boundary: {self.sample_id}")

    @property
    def completion_mask(self) -> list[bool]:
        return [index >= self.completion_start for index in range(len(self.input_ids))]


def read_valid_rollouts(
    path: str | Path, *, max_tokens: int = 16_384
) -> list[RolloutRecord]:
    records: list[RolloutRecord] = []
    seen_ids: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = RolloutRecord(**json.loads(line))
                record.validate(max_tokens=max_tokens)
                if record.sample_id in seen_ids:
                    raise ValueError(f"duplicate sample_id: {record.sample_id}")
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"invalid rollout at line {line_number}: {error}"
                ) from error
            seen_ids.add(record.sample_id)
            records.append(record)
    if not records:
        raise ValueError("rollout file contains no valid samples")
    return records


def shard_valid_rollouts(
    source: str | Path,
    output: str | Path,
    *,
    shard_size: int = 480,
    gradient_accumulation_steps: int = 8,
    max_tokens: int = 16_384,
) -> dict[str, Any]:
    """Freeze bounded rollout shards without splitting an accumulation group."""

    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    if shard_size % gradient_accumulation_steps:
        raise ValueError("shard_size must be divisible by gradient accumulation")
    records = read_valid_rollouts(source, max_tokens=max_tokens)
    target = Path(output)
    target.mkdir(parents=True, exist_ok=True)
    shards: list[dict[str, Any]] = []
    for shard_index, start in enumerate(range(0, len(records), shard_size)):
        shard_records = records[start : start + shard_size]
        filename = f"rollouts-{shard_index:04d}.jsonl"
        path = target / filename
        with path.open("w", encoding="utf-8") as handle:
            for record in shard_records:
                handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
        shards.append(
            {
                "index": shard_index,
                "file": filename,
                "samples": len(shard_records),
                "sample_ids": [record.sample_id for record in shard_records],
            }
        )
    source_digest = hashlib.sha256(Path(source).read_bytes()).hexdigest()
    plan: dict[str, Any] = {
        "version": "opd-rollout-shards-v1",
        "source_sha256": source_digest,
        "total_samples": len(records),
        "shard_size": shard_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "max_tokens": max_tokens,
        "shards": shards,
    }
    with (target / "plan.json").open("w", encoding="utf-8") as handle:
        json.dump(plan, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    return plan


def load_rollout_shard_plan(path: str | Path) -> tuple[dict[str, Any], str]:
    source = Path(path)
    raw = source.read_bytes()
    plan = json.loads(raw)
    if plan.get("version") != "opd-rollout-shards-v1":
        raise ValueError("unsupported rollout shard plan")
    shards = plan.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("rollout shard plan has no shards")
    if [shard.get("index") for shard in shards] != list(range(len(shards))):
        raise ValueError("rollout shard indices are not contiguous")
    sample_ids = [sample_id for shard in shards for sample_id in shard["sample_ids"]]
    if len(sample_ids) != plan.get("total_samples") or len(set(sample_ids)) != len(
        sample_ids
    ):
        raise ValueError("rollout shard plan sample IDs are inconsistent")
    return plan, hashlib.sha256(raw).hexdigest()


def save_teacher_artifact(
    destination: str | Path,
    *,
    record: RolloutRecord,
    hidden_states: torch.Tensor,
) -> None:
    from safetensors.torch import save_file

    record.validate()
    if hidden_states.ndim == 3 and hidden_states.shape[0] == 1:
        hidden_states = hidden_states[0]
    if hidden_states.ndim != 2 or hidden_states.shape[0] != len(record.input_ids):
        raise ValueError("teacher hidden states do not align with rollout tokens")
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {
        "input_ids": torch.tensor(record.input_ids, dtype=torch.int32),
        "completion_mask": torch.tensor(record.completion_mask, dtype=torch.bool),
        "teacher_hidden": hidden_states.detach()
        .to(device="cpu", dtype=torch.bfloat16)
        .contiguous(),
    }
    save_file(tensors, path, metadata={"sample_id": record.sample_id})
