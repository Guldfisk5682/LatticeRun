"""Shared serial-OPD input and device helpers."""

from __future__ import annotations

import json
from pathlib import Path

import torch


def cuda_device() -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this workflow")
    return "cuda"


def read_jsonl(path: str | Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"JSONL row {line_number} is not an object")
            rows.append(value)
    return rows


def load_clip_ratios(path: str | Path | None) -> dict[str, float]:
    if path is None:
        return {}
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return {str(key): float(value) for key, value in data.items()}
