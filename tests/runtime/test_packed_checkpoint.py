import json
from pathlib import Path

import numpy as np
from latticerun.cli import build_parser
from latticerun.runtime import (
    PACKED_FORMAT,
    pack_codes,
    pack_reference_checkpoint,
    unpack_codes,
    verify_packed_checkpoint,
)
from safetensors.numpy import save_file


def _reference(target: Path) -> None:
    target.mkdir()
    first = {
        "linear.weight.codes": np.array(
            [[[-3, -2, -1, 0], [1, 2, 3, -3]]], dtype=np.int8
        ),
        "linear.weight.scales": np.array([[[0.25], [0.5]]], dtype=np.float32),
        "norm.weight": np.array([1.0, 2.0], dtype=np.float16),
    }
    second = {
        "head.weight.codes": np.array([[-7, -1, 0, 1, 7]], dtype=np.int8),
        "head.weight.scales": np.array([[0.125]], dtype=np.float32),
    }
    save_file(first, target / "model-00001.safetensors")
    save_file(second, target / "model-00002.safetensors")
    index = {
        "format": "latticerun-effective-reference-v1",
        "packed_int3": False,
        "source_model": "test/model",
        "source_revision": "abc",
        "recovery": {"adapter_type": "dora", "mode": "ste"},
        "weight_map": {
            name: "model-00001.safetensors" for name in first
        }
        | {name: "model-00002.safetensors" for name in second},
        "quantization": {
            "linear.weight": {
                "bits": 3,
                "group_size": 4,
                "shape": [1, 8],
                "padded_columns": 0,
            },
            "head.weight": {
                "bits": 4,
                "group_size": 8,
                "shape": [1, 5],
                "padded_columns": 3,
            },
        },
    }
    (target / "model.latticerun.index.json").write_text(json.dumps(index))
    (target / "config.json").write_text('{"model_type":"test"}\n')


def test_code_pack_round_trip_for_tail_lengths():
    rng = np.random.default_rng(123)
    for bits, qmax in ((3, 3), (4, 7)):
        for count in range(35):
            codes = rng.integers(-qmax, qmax + 1, count, dtype=np.int8)
            payload = pack_codes(codes, bits=bits)
            assert len(payload) == (count * bits + 7) // 8
            assert np.array_equal(unpack_codes(payload, bits=bits, count=count), codes)


def test_runtime_cli_requires_explicit_artifact_paths():
    pack = build_parser().parse_args(
        ["runtime", "pack-reference", "--source", "reference", "--output", "packed"]
    )
    verify = build_parser().parse_args(
        ["runtime", "verify-packed", "--packed", "packed", "--source", "reference"]
    )
    assert (pack.source, pack.output) == ("reference", "packed")
    assert (verify.packed, verify.source) == ("packed", "reference")


def test_pack_and_full_source_verification(tmp_path):
    source = tmp_path / "reference"
    packed = tmp_path / "packed"
    _reference(source)

    result = pack_reference_checkpoint(source, packed)
    index = json.loads((packed / "model.latticerun.packed.index.json").read_text())
    report = verify_packed_checkpoint(packed, source=source)

    assert result["segments"] == 2
    assert index["format"] == PACKED_FORMAT
    assert index["packed_int3"] is True
    assert index["runtime_kernel_required"] is True
    assert index["low_vram_claim"] is False
    assert index["entries"]["linear.weight.codes"]["nbytes"] == 3
    assert index["entries"]["head.weight.codes"]["nbytes"] == 3
    assert report["status"] == "verified"
    assert report["tensors_verified"] == 5
    assert report["code_values_verified"] == 13
    assert (packed / "config.json").read_bytes() == (source / "config.json").read_bytes()
    assert not (packed / "INCOMPLETE").exists()

    transfer_report = verify_packed_checkpoint(packed)
    assert transfer_report["tensor_comparison"] is False
    assert (packed / "verification.json").exists()
    assert (packed / "transfer-verification.json").exists()

    # Completed segments are reusable at shard granularity.
    second = pack_reference_checkpoint(source, packed)
    assert second["packed_bytes"] == result["packed_bytes"]


def test_digest_verification_detects_corruption(tmp_path):
    source = tmp_path / "reference"
    packed = tmp_path / "packed"
    _reference(source)
    pack_reference_checkpoint(source, packed)
    segment = packed / "model-00001.lrpack"
    with segment.open("r+b") as handle:
        handle.seek(-1, 2)
        handle.write(b"\xff")
    try:
        verify_packed_checkpoint(packed)
    except RuntimeError as error:
        assert "digest mismatch" in str(error)
    else:
        raise AssertionError("corrupted packed segment was accepted")
