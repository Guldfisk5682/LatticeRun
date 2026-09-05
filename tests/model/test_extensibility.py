import subprocess
import sys
from pathlib import Path

import pytest
import torch
from latticerun.adapters import QuantLoRALinear, prepare_quantized_student
from latticerun.config import RecoveryConfig
from latticerun.model import (
    ModelSpec,
    ModuleDecision,
    register_model,
    register_model_adapter,
    resolve_model,
    resolve_model_adapter,
)


class _FakeAdapter:
    name = "fake"

    def inspect_named_modules(
        self, named_modules, *, enable_deltanet_z=False, strict=True
    ):
        del named_modules, enable_deltanet_z, strict
        return [
            ModuleDecision(
                name="0",
                role="ffn",
                bits=3,
                compute_dtype="bf16",
                opd_eligible=True,
                reason="test policy",
            )
        ]

    def load_model(self, *args, **kwargs):
        raise NotImplementedError

    def render_prompt(
        self, tokenizer, row, *, thinking, reasoning_effort, force_chat_template
    ):
        del tokenizer, thinking, reasoning_effort, force_chat_template
        return str(row["prompt"])

    def block_name_for_module(self, module_name):
        return module_name.split(".", 1)[0]

    def forward_hidden(self, model, **model_inputs):
        return model(**model_inputs)


def test_generic_recovery_accepts_an_injected_non_qwen_model_adapter():
    model = torch.nn.Sequential(torch.nn.Linear(8, 6, bias=False))
    replaced = prepare_quantized_student(
        model,
        RecoveryConfig(adapter_type="lora", rank=2, alpha=4, mode="ste"),
        _FakeAdapter(),
        group_size=4,
    )
    assert replaced == ["0"]
    assert isinstance(model[0], QuantLoRALinear)


def test_generic_layers_do_not_import_concrete_model_implementations():
    source_root = Path(__file__).parents[2] / "src"
    generic_files = [
        *(source_root / "quant").glob("*.py"),
        *(source_root / "adapters").glob("*.py"),
        *(source_root / "opd").glob("*.py"),
    ]
    for path in generic_files:
        source = path.read_text(encoding="utf-8")
        assert "qwen35" not in source.lower(), path
        assert "QWEN35_ADAPTER" not in source, path


def test_registry_resolves_the_current_adapter_at_the_orchestration_boundary():
    assert resolve_model_adapter("qwen35").name == "qwen35"


def test_registry_resolves_qwen38_as_the_default_public_model():
    model_id, adapter = resolve_model("qwen3.8-27b")
    assert model_id == "Qwen/Qwen3.8-27B"
    assert adapter.name == "qwen35"


def test_unregistered_hub_models_require_an_explicit_adapter():
    with pytest.raises(ValueError, match="explicit model adapter"):
        resolve_model("organization/future-model")


def test_registry_accepts_future_model_adapters_without_generic_code_changes():
    adapter = _FakeAdapter()
    register_model_adapter(adapter)
    register_model(
        ModelSpec(
            name="test-future-model",
            model_id="organization/future-model",
            adapter=adapter.name,
        )
    )
    assert resolve_model_adapter("fake") is adapter
    assert resolve_model("test-future-model") == (
        "organization/future-model",
        adapter,
    )


def test_importing_generic_packages_does_not_load_qwen_implementation():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import latticerun.adapters, latticerun.quant, latticerun.opd; "
                "assert 'latticerun.model.qwen35' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
