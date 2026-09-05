"""Lossless bit-packing for LatticeRun mixed INT3/INT4 reference checkpoints.

The reference exporter deliberately stores quantized codes as int8 tensors so
they can be audited.  This module turns that representation into a compact,
runtime-oriented payload without quantizing weights a second time.  It does not
claim compatibility with Transformers or vLLM: a custom loader/kernel is still
required to consume the packed code streams directly.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import time
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import BinaryIO

import numpy as np

REFERENCE_INDEX = "model.latticerun.index.json"
PACKED_INDEX = "model.latticerun.packed.index.json"
VERIFICATION_REPORT = "verification.json"
TRANSFER_VERIFICATION_REPORT = "transfer-verification.json"
PACKED_FORMAT = "latticerun-mixed-q3-q4-packed-v1"
ALIGNMENT = 64
_COPY_CHUNK_BYTES = 64 * 1024 * 1024
_CODE_CHUNK_VALUES = 64 * 1024 * 1024

_DTYPE_BYTES = {
    "BOOL": 1,
    "I8": 1,
    "U8": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "F64": 8,
    "I64": 8,
    "U64": 8,
}


@dataclass(frozen=True, slots=True)
class SafeTensor:
    name: str
    dtype: str
    shape: tuple[int, ...]
    offset: int
    nbytes: int

    @property
    def numel(self) -> int:
        result = 1
        for value in self.shape:
            result *= value
        return result


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_COPY_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _product(shape: tuple[int, ...]) -> int:
    result = 1
    for value in shape:
        if value < 0:
            raise ValueError(f"negative tensor dimension: {shape}")
        result *= value
    return result


def _read_safetensors_header(path: Path) -> dict[str, SafeTensor]:
    with path.open("rb") as handle:
        length_bytes = handle.read(8)
        if len(length_bytes) != 8:
            raise ValueError(f"truncated safetensors header: {path}")
        header_length = struct.unpack("<Q", length_bytes)[0]
        raw_header = handle.read(header_length)
        if len(raw_header) != header_length:
            raise ValueError(f"truncated safetensors JSON header: {path}")
    header = json.loads(raw_header)
    data_start = 8 + header_length
    file_size = path.stat().st_size
    tensors: dict[str, SafeTensor] = {}
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        dtype = spec["dtype"]
        shape = tuple(int(value) for value in spec["shape"])
        relative_start, relative_end = map(int, spec["data_offsets"])
        if dtype not in _DTYPE_BYTES:
            raise ValueError(f"unsupported safetensors dtype {dtype!r}: {name}")
        expected = _product(shape) * _DTYPE_BYTES[dtype]
        if relative_end - relative_start != expected:
            raise ValueError(f"invalid byte extent for tensor {name}")
        start = data_start + relative_start
        end = data_start + relative_end
        if not 0 <= start <= end <= file_size:
            raise ValueError(f"tensor outside safetensors file: {name}")
        tensors[name] = SafeTensor(name, dtype, shape, start, expected)
    return tensors


def _pack_q3(values: np.ndarray) -> bytes:
    signed = np.asarray(values, dtype=np.int8).reshape(-1)
    if signed.size and (signed.min() < -3 or signed.max() > 3):
        raise ValueError("INT3 codes exceed symmetric [-3, 3] range")
    count = signed.size
    padded = np.zeros((-count) % 8, dtype=np.uint8)
    unsigned = (signed.astype(np.int16) + 3).astype(np.uint8)
    if padded.size:
        unsigned = np.concatenate((unsigned, padded))
    groups = unsigned.reshape(-1, 8).astype(np.uint16)
    output = np.empty((groups.shape[0], 3), dtype=np.uint8)
    output[:, 0] = groups[:, 0] | (groups[:, 1] << 3) | ((groups[:, 2] & 3) << 6)
    output[:, 1] = (
        (groups[:, 2] >> 2)
        | (groups[:, 3] << 1)
        | (groups[:, 4] << 4)
        | ((groups[:, 5] & 1) << 7)
    )
    output[:, 2] = (groups[:, 5] >> 1) | (groups[:, 6] << 2) | (groups[:, 7] << 5)
    return output.reshape(-1)[: (count * 3 + 7) // 8].tobytes()


def _unpack_q3(payload: bytes, count: int) -> np.ndarray:
    if count < 0:
        raise ValueError("count must be non-negative")
    expected = (count * 3 + 7) // 8
    if len(payload) != expected:
        raise ValueError(f"INT3 payload is {len(payload)} bytes, expected {expected}")
    raw = np.frombuffer(payload, dtype=np.uint8)
    padded = np.zeros((-raw.size) % 3, dtype=np.uint8)
    if padded.size:
        raw = np.concatenate((raw, padded))
    groups = raw.reshape(-1, 3).astype(np.uint16)
    values = np.empty((groups.shape[0], 8), dtype=np.uint8)
    values[:, 0] = groups[:, 0] & 7
    values[:, 1] = (groups[:, 0] >> 3) & 7
    values[:, 2] = ((groups[:, 0] >> 6) & 3) | ((groups[:, 1] & 1) << 2)
    values[:, 3] = (groups[:, 1] >> 1) & 7
    values[:, 4] = (groups[:, 1] >> 4) & 7
    values[:, 5] = ((groups[:, 1] >> 7) & 1) | ((groups[:, 2] & 3) << 1)
    values[:, 6] = (groups[:, 2] >> 2) & 7
    values[:, 7] = (groups[:, 2] >> 5) & 7
    return (values.reshape(-1)[:count].astype(np.int16) - 3).astype(np.int8)


def _pack_q4(values: np.ndarray) -> bytes:
    signed = np.asarray(values, dtype=np.int8).reshape(-1)
    if signed.size and (signed.min() < -7 or signed.max() > 7):
        raise ValueError("INT4 codes exceed symmetric [-7, 7] range")
    count = signed.size
    unsigned = (signed.astype(np.int16) + 7).astype(np.uint8)
    if count % 2:
        unsigned = np.concatenate((unsigned, np.zeros(1, dtype=np.uint8)))
    pairs = unsigned.reshape(-1, 2)
    output = pairs[:, 0] | (pairs[:, 1] << 4)
    return output.tobytes()


def _unpack_q4(payload: bytes, count: int) -> np.ndarray:
    if count < 0:
        raise ValueError("count must be non-negative")
    expected = (count + 1) // 2
    if len(payload) != expected:
        raise ValueError(f"INT4 payload is {len(payload)} bytes, expected {expected}")
    raw = np.frombuffer(payload, dtype=np.uint8)
    values = np.empty(raw.size * 2, dtype=np.uint8)
    values[0::2] = raw & 15
    values[1::2] = raw >> 4
    return (values[:count].astype(np.int16) - 7).astype(np.int8)


def pack_codes(values: np.ndarray, *, bits: int) -> bytes:
    """Pack signed symmetric Q3/Q4 codes in row-major LSB-first order."""

    if bits == 3:
        return _pack_q3(values)
    if bits == 4:
        return _pack_q4(values)
    raise ValueError("packed checkpoints support only INT3 and INT4 codes")


def unpack_codes(payload: bytes, *, bits: int, count: int) -> np.ndarray:
    """Inverse of :func:`pack_codes`, primarily for audits and loader tests."""

    if bits == 3:
        return _unpack_q3(payload, count)
    if bits == 4:
        return _unpack_q4(payload, count)
    raise ValueError("packed checkpoints support only INT3 and INT4 codes")


def _iter_file_range(
    handle: BinaryIO, offset: int, length: int, chunk_size: int
) -> Iterator[bytes]:
    handle.seek(offset)
    remaining = length
    while remaining:
        chunk = handle.read(min(remaining, chunk_size))
        if not chunk:
            raise EOFError("unexpected end of source tensor")
        remaining -= len(chunk)
        yield chunk


def _align(handle: BinaryIO) -> int:
    padding = (-handle.tell()) % ALIGNMENT
    if padding:
        handle.write(bytes(padding))
    return handle.tell()


def _packed_nbytes(count: int, bits: int) -> int:
    return (count * bits + 7) // 8


def _copy_auxiliary_files(source: Path, target: Path) -> list[str]:
    copied: list[str] = []
    excluded = {REFERENCE_INDEX, PACKED_INDEX, VERIFICATION_REPORT, "INCOMPLETE"}
    for path in sorted(source.iterdir()):
        if not path.is_file() or path.name in excluded or path.suffix == ".safetensors":
            continue
        destination = target / path.name
        temporary = target / f".{path.name}.tmp"
        shutil.copy2(path, temporary)
        os.replace(temporary, destination)
        copied.append(path.name)
    return copied


def _source_layout(
    source: Path, reference: dict[str, object]
) -> tuple[dict[str, tuple[Path, SafeTensor]], dict[str, list[SafeTensor]]]:
    weight_map = reference.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("reference index has no weight_map")
    headers: dict[str, dict[str, SafeTensor]] = {}
    by_shard: dict[str, list[SafeTensor]] = defaultdict(list)
    result: dict[str, tuple[Path, SafeTensor]] = {}
    for name, filename_value in weight_map.items():
        filename = str(filename_value)
        path = source / filename
        if filename not in headers:
            headers[filename] = _read_safetensors_header(path)
        try:
            tensor = headers[filename][name]
        except KeyError as error:
            raise KeyError(f"{name} is absent from {filename}") from error
        result[name] = (path, tensor)
        by_shard[filename].append(tensor)
    for filename, tensors in by_shard.items():
        by_shard[filename] = sorted(tensors, key=lambda item: item.offset)
    return result, dict(by_shard)


def _pack_segment(
    source_path: Path,
    tensors: list[SafeTensor],
    destination: Path,
    quantization: dict[str, dict[str, object]],
) -> dict[str, object]:
    temporary = destination.with_suffix(destination.suffix + ".partial")
    entries: dict[str, dict[str, object]] = {}
    source_digest = hashlib.sha256()
    with source_path.open("rb") as source_handle, temporary.open("wb") as output:
        for tensor in tensors:
            output_offset = _align(output)
            digest = hashlib.sha256()
            if tensor.name.endswith(".codes"):
                prefix = tensor.name[: -len(".codes")]
                try:
                    metadata = quantization[prefix]
                except KeyError as error:
                    raise KeyError(f"missing quantization metadata for {prefix}") from error
                bits = int(metadata["bits"])
                if bits not in (3, 4):
                    raise ValueError(f"unsupported code width {bits}: {tensor.name}")
                if tensor.dtype != "I8":
                    raise ValueError(f"codes must use I8 source storage: {tensor.name}")
                unit = 8 if bits == 3 else 2
                chunk_values = (_CODE_CHUNK_VALUES // unit) * unit
                remaining = tensor.numel
                source_handle.seek(tensor.offset)
                while remaining:
                    count = min(remaining, chunk_values)
                    if count < remaining:
                        count -= count % unit
                    raw = source_handle.read(count)
                    if len(raw) != count:
                        raise EOFError(f"truncated source tensor: {tensor.name}")
                    source_digest.update(raw)
                    packed = pack_codes(np.frombuffer(raw, dtype=np.int8), bits=bits)
                    output.write(packed)
                    digest.update(packed)
                    remaining -= count
                output_nbytes = _packed_nbytes(tensor.numel, bits)
                entries[tensor.name] = {
                    "bits": bits,
                    "encoding": "biased-symmetric-lsb-first",
                    "file": destination.name,
                    "numel": tensor.numel,
                    "nbytes": output_nbytes,
                    "offset": output_offset,
                    "packed_sha256": digest.hexdigest(),
                    "qmax": (1 << (bits - 1)) - 1,
                    "shape": list(tensor.shape),
                    "source_dtype": tensor.dtype,
                }
            else:
                for chunk in _iter_file_range(
                    source_handle, tensor.offset, tensor.nbytes, _COPY_CHUNK_BYTES
                ):
                    source_digest.update(chunk)
                    digest.update(chunk)
                    output.write(chunk)
                entries[tensor.name] = {
                    "encoding": "raw",
                    "file": destination.name,
                    "nbytes": tensor.nbytes,
                    "offset": output_offset,
                    "raw_sha256": digest.hexdigest(),
                    "shape": list(tensor.shape),
                    "source_dtype": tensor.dtype,
                }
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, destination)
    return {
        "entries": entries,
        "output_file": destination.name,
        "output_sha256": _sha256(destination),
        "output_size": destination.stat().st_size,
        "source_data_sha256": source_digest.hexdigest(),
        "source_file": source_path.name,
        "source_mtime_ns": source_path.stat().st_mtime_ns,
        "source_size": source_path.stat().st_size,
    }


def pack_reference_checkpoint(
    source: str | Path, target: str | Path
) -> dict[str, object]:
    """Pack a reference checkpoint, resuming at completed shard boundaries."""

    started = time.perf_counter()
    source_path = Path(source).resolve()
    target_path = Path(target).resolve()
    if source_path == target_path:
        raise ValueError("source and target must be different directories")
    reference_path = source_path / REFERENCE_INDEX
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    if reference.get("format") != "latticerun-effective-reference-v1":
        raise ValueError("source is not a LatticeRun effective reference checkpoint")
    if reference.get("packed_int3") is not False:
        raise ValueError("source reference unexpectedly claims packed INT3 storage")
    target_path.mkdir(parents=True, exist_ok=True)
    incomplete = target_path / "INCOMPLETE"
    incomplete.write_text("packing in progress\n", encoding="utf-8")
    state_path = target_path / ".pack-state"
    state_path.mkdir(exist_ok=True)

    _, by_shard = _source_layout(source_path, reference)
    quantization = reference.get("quantization")
    if not isinstance(quantization, dict):
        raise TypeError("reference index has no quantization metadata")
    all_entries: dict[str, dict[str, object]] = {}
    segments: list[dict[str, object]] = []
    for number, (filename, tensors) in enumerate(sorted(by_shard.items()), start=1):
        source_shard = source_path / filename
        output_shard = target_path / f"model-{number:05d}.lrpack"
        state_file = state_path / f"model-{number:05d}.json"
        state: dict[str, object] | None = None
        if state_file.exists() and output_shard.exists():
            candidate = json.loads(state_file.read_text(encoding="utf-8"))
            stat = source_shard.stat()
            if (
                candidate.get("source_file") == filename
                and candidate.get("source_size") == stat.st_size
                and candidate.get("source_mtime_ns") == stat.st_mtime_ns
                and candidate.get("output_size") == output_shard.stat().st_size
                and candidate.get("output_sha256") == _sha256(output_shard)
            ):
                state = candidate
        if state is None:
            state = _pack_segment(
                source_shard, tensors, output_shard, quantization
            )
            _atomic_json(state_file, state)
        entries = state["entries"]
        if not isinstance(entries, dict):
            raise TypeError(f"invalid pack state: {state_file}")
        overlap = set(all_entries).intersection(entries)
        if overlap:
            raise ValueError(f"duplicate packed tensors: {sorted(overlap)[:3]}")
        all_entries.update(entries)
        segments.append({key: value for key, value in state.items() if key != "entries"})

    source_names = set(reference["weight_map"])
    if set(all_entries) != source_names:
        missing = source_names - set(all_entries)
        unexpected = set(all_entries) - source_names
        raise RuntimeError(
            f"packed tensor map mismatch: missing={len(missing)}, unexpected={len(unexpected)}"
        )
    copied = _copy_auxiliary_files(source_path, target_path)
    packed_index = {
        "alignment_bytes": ALIGNMENT,
        "auxiliary_files": copied,
        "entries": all_entries,
        "format": PACKED_FORMAT,
        "low_vram_claim": False,
        "packed_int3": True,
        "quantization": quantization,
        "recovery": reference.get("recovery"),
        "runtime_kernel_required": True,
        "segments": segments,
        "source_format": reference["format"],
        "source_model": reference.get("source_model"),
        "source_revision": reference.get("source_revision"),
    }
    _atomic_json(target_path / PACKED_INDEX, packed_index)
    incomplete.unlink(missing_ok=True)
    total_size = sum(path.stat().st_size for path in target_path.glob("*.lrpack"))
    return {
        "output": str(target_path),
        "packed_bytes": total_size,
        "segments": len(segments),
        "tensors": len(all_entries),
        "wall_seconds": time.perf_counter() - started,
    }


def _compare_raw(
    source_handle: BinaryIO,
    source_tensor: SafeTensor,
    packed_handle: BinaryIO,
    packed_offset: int,
) -> int:
    source_handle.seek(source_tensor.offset)
    packed_handle.seek(packed_offset)
    remaining = source_tensor.nbytes
    while remaining:
        count = min(remaining, _COPY_CHUNK_BYTES)
        source_chunk = source_handle.read(count)
        packed_chunk = packed_handle.read(count)
        if source_chunk != packed_chunk:
            raise RuntimeError(f"raw tensor differs after packing: {source_tensor.name}")
        remaining -= count
    return source_tensor.nbytes


def _compare_codes(
    source_handle: BinaryIO,
    source_tensor: SafeTensor,
    packed_handle: BinaryIO,
    packed_offset: int,
    bits: int,
) -> int:
    source_handle.seek(source_tensor.offset)
    packed_handle.seek(packed_offset)
    unit = 8 if bits == 3 else 2
    chunk_values = (_CODE_CHUNK_VALUES // unit) * unit
    remaining = source_tensor.numel
    while remaining:
        count = min(remaining, chunk_values)
        if count < remaining:
            count -= count % unit
        packed_count = _packed_nbytes(count, bits)
        source_raw = source_handle.read(count)
        packed_raw = packed_handle.read(packed_count)
        if len(source_raw) != count or len(packed_raw) != packed_count:
            raise EOFError(f"truncated tensor during verification: {source_tensor.name}")
        restored = unpack_codes(packed_raw, bits=bits, count=count)
        if not np.array_equal(restored, np.frombuffer(source_raw, dtype=np.int8)):
            raise RuntimeError(f"codes differ after packing: {source_tensor.name}")
        remaining -= count
    return source_tensor.numel


def verify_packed_checkpoint(
    packed: str | Path, *, source: str | Path | None = None
) -> dict[str, object]:
    """Verify segment digests and, when supplied, every tensor against source."""

    started = time.perf_counter()
    packed_path = Path(packed).resolve()
    if (packed_path / "INCOMPLETE").exists():
        raise RuntimeError("packed checkpoint is marked INCOMPLETE")
    index = json.loads((packed_path / PACKED_INDEX).read_text(encoding="utf-8"))
    if index.get("format") != PACKED_FORMAT:
        raise ValueError("unsupported packed checkpoint format")
    entries = index.get("entries")
    segments = index.get("segments")
    if not isinstance(entries, dict) or not isinstance(segments, list):
        raise TypeError("packed index is missing entries or segments")
    segment_sizes: dict[str, int] = {}
    for segment in segments:
        path = packed_path / segment["output_file"]
        if path.stat().st_size != segment["output_size"]:
            raise RuntimeError(f"packed segment size mismatch: {path.name}")
        if _sha256(path) != segment["output_sha256"]:
            raise RuntimeError(f"packed segment digest mismatch: {path.name}")
        segment_sizes[path.name] = path.stat().st_size
    extents: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    for name, entry in entries.items():
        filename = entry.get("file")
        if filename not in segment_sizes:
            raise RuntimeError(f"tensor references an unknown segment: {name}")
        offset = int(entry["offset"])
        nbytes = int(entry["nbytes"])
        if offset % ALIGNMENT or nbytes < 0 or offset + nbytes > segment_sizes[filename]:
            raise RuntimeError(f"invalid packed extent: {name}")
        if entry.get("encoding") == "biased-symmetric-lsb-first":
            expected = _packed_nbytes(int(entry["numel"]), int(entry["bits"]))
            if nbytes != expected:
                raise RuntimeError(f"invalid packed code size: {name}")
        elif entry.get("encoding") != "raw":
            raise RuntimeError(f"unknown tensor encoding: {name}")
        extents[filename].append((offset, offset + nbytes, name))
    for filename, ranges in extents.items():
        ranges.sort()
        for previous, current in pairwise(ranges):
            if previous[1] > current[0]:
                raise RuntimeError(
                    f"overlapping tensors in {filename}: {previous[2]}, {current[2]}"
                )

    report: dict[str, object] = {
        "format": PACKED_FORMAT,
        "packed": str(packed_path),
        "segments_verified": len(segments),
        "status": "verified",
        "tensor_comparison": source is not None,
        "tensors_verified": 0,
    }
    if source is not None:
        source_path = Path(source).resolve()
        reference = json.loads((source_path / REFERENCE_INDEX).read_text(encoding="utf-8"))
        source_tensors, _ = _source_layout(source_path, reference)
        if set(entries) != set(source_tensors):
            raise RuntimeError("packed and source tensor sets differ")
        handles: dict[Path, BinaryIO] = {}
        packed_handles: dict[Path, BinaryIO] = {}
        code_values = 0
        raw_bytes = 0
        try:
            for name in sorted(entries):
                entry = entries[name]
                source_file, source_tensor = source_tensors[name]
                source_handle = handles.get(source_file)
                if source_handle is None:
                    source_handle = source_file.open("rb")
                    handles[source_file] = source_handle
                packed_file = packed_path / entry["file"]
                packed_handle = packed_handles.get(packed_file)
                if packed_handle is None:
                    packed_handle = packed_file.open("rb")
                    packed_handles[packed_file] = packed_handle
                if entry["encoding"] == "raw":
                    raw_bytes += _compare_raw(
                        source_handle, source_tensor, packed_handle, int(entry["offset"])
                    )
                else:
                    code_values += _compare_codes(
                        source_handle,
                        source_tensor,
                        packed_handle,
                        int(entry["offset"]),
                        int(entry["bits"]),
                    )
                report["tensors_verified"] = int(report["tensors_verified"]) + 1
        finally:
            for handle in (*handles.values(), *packed_handles.values()):
                handle.close()
        report["code_values_verified"] = code_values
        report["raw_bytes_verified"] = raw_bytes
        report["source"] = str(source_path)
    report["wall_seconds"] = time.perf_counter() - started
    report_name = VERIFICATION_REPORT if source is not None else TRANSFER_VERIFICATION_REPORT
    _atomic_json(packed_path / report_name, report)
    return report


__all__ = [
    "PACKED_FORMAT",
    "PACKED_INDEX",
    "pack_codes",
    "pack_reference_checkpoint",
    "unpack_codes",
    "verify_packed_checkpoint",
]
