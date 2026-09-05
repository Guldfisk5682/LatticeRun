import json

import pytest
from latticerun.cli import build_parser
from latticerun.config import (
    DEFAULT_AWQ_CLIP_RATIOS,
    DEFAULT_MODEL_NAME,
    AWQConfig,
    OPDConfig,
    ProjectConfig,
    RecoveryConfig,
    load_config,
    save_config,
)
from latticerun.model import QWEN35_ADAPTER


class _Tokenizer:
    def __init__(self):
        self.kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.kwargs = kwargs
        return str(messages)


def test_qwen38_is_the_default_model_and_recovery_recipe():
    config = ProjectConfig()
    assert config.model == DEFAULT_MODEL_NAME == "qwen3.8-27b"
    assert config.recovery == RecoveryConfig(adapter_type="dora", mode="ste")
    assert not hasattr(config.recovery, "dropout")

    args = build_parser().parse_args(
        ["opd", "rollout", "--prompts", "p.jsonl", "--output", "r.jsonl"]
    )
    assert args.model == "qwen3.8-27b"
    assert args.model_adapter is None
    assert args.adapter_type == "dora"
    assert args.mode == "ste"
    assert args.thinking is True
    assert args.reasoning_effort == "medium"
    assert args.do_sample is True
    assert args.temperature == 1.0
    assert args.top_p == 0.95
    assert args.top_k == 20


def test_rollout_accepts_the_validated_16k_ceiling():
    args = build_parser().parse_args(
        [
            "opd",
            "rollout",
            "--prompts",
            "p.jsonl",
            "--output",
            "r.jsonl",
            "--max-tokens",
            "16384",
            "--max-new-tokens",
            "16384",
        ]
    )
    assert args.max_tokens == 16_384
    assert args.max_new_tokens == 16_384


def test_reference_export_requires_an_explicit_destination():
    args = build_parser().parse_args(
        [
            "opd",
            "export",
            "--adapter",
            "adapter",
            "--clip-ratios",
            "ratios.json",
            "--output",
            "reference",
        ]
    )
    assert args.output == "reference"
    assert args.max_shard_bytes == 3_500_000_000


def test_experiment_and_runtime_commands_are_not_on_main():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["eval", "preflight"])
    with pytest.raises(SystemExit):
        parser.parse_args(["runtime", "verify-packed"])


def test_rollout_prompt_explicitly_enables_thinking():
    tokenizer = _Tokenizer()
    QWEN35_ADAPTER.render_prompt(
        tokenizer,
        {"prompt": "solve this"},
        thinking=True,
        reasoning_effort="medium",
        force_chat_template=True,
    )
    assert tokenizer.kwargs["enable_thinking"] is True
    assert tokenizer.kwargs["reasoning_effort"] == "medium"
    assert tokenizer.kwargs["add_generation_prompt"] is True


def test_opd_training_defaults_match_the_validated_recipe():
    args = build_parser().parse_args(
        [
            "opd",
            "train",
            "--teacher-artifacts",
            "teacher",
            "--output",
            "adapter",
        ]
    )
    assert args.logging_flush_secs == 10
    assert args.adapter_rank == 8
    assert args.adapter_alpha == 16
    assert args.learning_rate == 1e-4
    assert args.weight_decay == 0
    assert args.warmup_ratio == 0.03
    assert args.lr_scheduler_type == "cosine"
    assert args.gradient_accumulation_steps == 8
    assert args.max_grad_norm == 1.0
    assert args.checkpoint_interval_fraction == pytest.approx(1 / 3)
    assert args.save_total_limit == 1
    assert args.resume_from_checkpoint is None

    config = OPDConfig()
    assert config.temperature == 1.0
    assert config.learning_rate == 1e-4
    assert config.weight_decay == 0
    assert config.warmup_ratio == 0.03
    assert config.lr_scheduler_type == "cosine"
    assert config.gradient_accumulation_steps == 8
    assert config.max_grad_norm == 1.0
    assert config.checkpoint_interval_fraction == pytest.approx(1 / 3)
    assert config.save_total_limit == 1


def test_block_sensitivity_defaults_to_eight_high_and_two_middle_targets():
    args = build_parser().parse_args(
        [
            "sensitivity",
            "block",
            "--calibration-jsonl",
            "calibration.jsonl",
            "--awq-diagnostics",
            "diagnostics.json",
            "--output",
            "block.json",
        ]
    )
    assert args.high_count == 8
    assert args.middle_count == 2
    assert args.max_samples == 1
    assert args.max_tokens == 512


def test_awq_ratio_grid_is_the_default_but_remains_configurable():
    assert AWQConfig().clip_ratios == DEFAULT_AWQ_CLIP_RATIOS
    assert AWQConfig(clip_ratios=[1.0, 0.8]).clip_ratios == (1.0, 0.8)
    with pytest.raises(ValueError, match="must not be empty"):
        AWQConfig(clip_ratios=())
    with pytest.raises(ValueError, match=r"in \(0, 1\]"):
        AWQConfig(clip_ratios=(1.1,))


def test_default_project_config_json_roundtrip(tmp_path):
    path = tmp_path / "config.json"
    expected = ProjectConfig()
    save_config(expected, path)

    serialized = json.loads(path.read_text(encoding="utf-8"))
    assert serialized["model"] == "qwen3.8-27b"
    assert isinstance(serialized["quant"]["awq"]["clip_ratios"], list)
    observed = load_config(path)

    assert observed == expected
    assert isinstance(observed.quant.awq.clip_ratios, tuple)


def test_legacy_model_id_and_dropout_are_read_without_restoring_old_fields(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "model_id": "Qwen/Qwen3.8-27B",
                "recovery": {"adapter_type": "lora", "dropout": 0.1},
            }
        ),
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.model == "Qwen/Qwen3.8-27B"
    assert config.recovery.adapter_type == "lora"
    assert not hasattr(config.recovery, "dropout")
