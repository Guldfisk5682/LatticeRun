import copy
import json
from pathlib import Path

import torch
from latticerun.adapters import (
    FrozenQuantEmbedding,
    FrozenQuantLinear,
    QuantDoRALinear,
    merge_recovery_for_inference,
    prepare_quantized_student,
    save_adapter,
)
from latticerun.config import RecoveryConfig
from latticerun.model import ModuleDecision
from latticerun.opd.export import ReferenceExportRequest, export_reference_student
from latticerun.opd.student import StudentRequest
from safetensors.torch import load_file


class _Config:
    def save_pretrained(self, target):
        Path(target, "config.json").write_text('{"model_type":"test"}\n')


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 6, bias=True, dtype=torch.bfloat16)
        self.embedding = torch.nn.Embedding(12, 8, dtype=torch.bfloat16)
        self.norm = torch.nn.LayerNorm(8, dtype=torch.bfloat16)
        self.config = _Config()


class _ModelAdapter:
    name = "test"

    def __init__(self, model):
        self.model = copy.deepcopy(model)

    def load_model(self, *args, **kwargs):
        del args, kwargs
        return copy.deepcopy(self.model)

    def inspect_named_modules(self, named_modules, **kwargs):
        del named_modules, kwargs
        return [
            ModuleDecision("linear", "ffn", 3, "bf16", True, "test recovery"),
            ModuleDecision("embedding", "embedding", 3, "bf16", False, "test quant"),
        ]

    def forward_hidden(self, model, **model_inputs):
        return model(**model_inputs)


class _Tokenizer:
    def save_pretrained(self, target):
        Path(target, "tokenizer_config.json").write_text("{}\n")


def _tensor(target, index, name):
    filename = index["weight_map"][name]
    return load_file(target / filename)[name]


def _dequantize(target, index, prefix):
    metadata = index["quantization"][prefix]
    codes = _tensor(target, index, f"{prefix}.codes")
    scales = _tensor(target, index, f"{prefix}.scales")
    rows, columns = metadata["shape"]
    return (codes.float() * scales).reshape(rows, -1)[:, :columns].to(torch.bfloat16)


def test_reference_export_is_exact_post_dora_quantization(monkeypatch, tmp_path):
    torch.manual_seed(7)
    base = _Model()
    adapter = _ModelAdapter(base)
    recovery = RecoveryConfig(adapter_type="dora", rank=2, alpha=4, mode="ste")
    ratios = {"linear": 0.8, "embedding": 0.9}
    ratios_path = tmp_path / "ratios.json"
    ratios_path.write_text(json.dumps(ratios))

    trained = copy.deepcopy(base)
    prepare_quantized_student(
        trained,
        recovery,
        adapter,
        group_size=4,
        clip_ratios=ratios,
    )
    assert isinstance(trained.linear, QuantDoRALinear)
    trained.linear.lora_a.data.normal_(std=0.05)
    trained.linear.lora_b.data.normal_(std=0.05)
    trained.linear.magnitude.data.mul_(1.03)
    adapter_path = tmp_path / "adapter"
    save_adapter(trained, adapter_path, recovery)
    merge_recovery_for_inference(trained)
    assert isinstance(trained.linear, FrozenQuantLinear)
    assert isinstance(trained.embedding, FrozenQuantEmbedding)

    monkeypatch.setattr("latticerun.opd.export.cuda_device", lambda: "cpu")
    monkeypatch.setattr(
        "latticerun.opd.export.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: _Tokenizer(),
    )
    target = tmp_path / "reference"
    result = export_reference_student(
        ReferenceExportRequest(
            student=StudentRequest(
                model="test-model",
                recovery=recovery,
                group_size=4,
                clip_ratios=ratios_path,
                adapter_checkpoint=adapter_path,
            ),
            output=target,
            max_shard_bytes=200,
        ),
        adapter,
    )

    index = json.loads((target / "model.latticerun.index.json").read_text())
    assert result["quantized_modules"] == 2
    assert index["format"] == "latticerun-effective-reference-v1"
    assert index["packed_int3"] is False
    assert index["quantization"]["linear.weight"]["opd_merged"] is True
    assert _tensor(target, index, "linear.weight.scales").dtype == torch.float32
    assert torch.equal(
        _dequantize(target, index, "linear.weight"), trained.linear.weight
    )
    assert torch.equal(
        _dequantize(target, index, "embedding.weight"), trained.embedding.weight
    )
    assert torch.equal(_tensor(target, index, "norm.weight"), trained.norm.weight)
    assert not (target / "INCOMPLETE").exists()
