"""Command-line entry points for quantization and recovery workflows."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict

from .config import DEFAULT_MODEL_NAME, ProjectConfig, RecoveryConfig, save_config


def cmd_init(args: argparse.Namespace) -> None:
    save_config(ProjectConfig(), args.output)
    print(args.output)


def cmd_doctor(args: argparse.Namespace) -> None:
    del args
    import torch
    import transformers

    print(
        json.dumps(
            {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda": torch.version.cuda,
                "gpu": (
                    torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
                ),
            },
            indent=2,
        )
    )


def _resolved_model(args: argparse.Namespace):
    from .model import resolve_model

    return resolve_model(args.model, args.model_adapter)


def _recovery(args: argparse.Namespace) -> RecoveryConfig:
    return RecoveryConfig(
        adapter_type=args.adapter_type,
        rank=args.adapter_rank,
        alpha=args.adapter_alpha,
        mode=args.mode,
    )


def _student(args: argparse.Namespace):
    from .opd.student import StudentRequest

    model_id, _ = _resolved_model(args)
    return StudentRequest(
        model=model_id,
        revision=args.revision,
        attn_implementation=args.attn_implementation,
        recovery=_recovery(args),
        group_size=args.group_size,
        clip_ratios=args.clip_ratios,
        adapter_checkpoint=args.adapter,
        enable_deltanet_z=args.enable_deltanet_z,
    )


def cmd_inspect(args: argparse.Namespace) -> None:
    import torch

    model_id, model_adapter = _resolved_model(args)
    model = model_adapter.load_model(
        model_id,
        revision=args.revision,
        dtype=torch.bfloat16,
        device_map={"": args.device},
        attn_implementation=args.attn_implementation,
    )
    decisions = model_adapter.inspect_named_modules(
        model.named_modules(), enable_deltanet_z=args.enable_deltanet_z, strict=True
    )
    print(json.dumps([asdict(item) for item in decisions], indent=2, default=str))


def cmd_quantize(args: argparse.Namespace) -> None:
    from .quant.workflow import QuantizeRequest, run_quantization

    model_id, model_adapter = _resolved_model(args)
    run_quantization(
        QuantizeRequest(
            model=model_id,
            revision=args.revision,
            attn_implementation=args.attn_implementation,
            calibration_jsonl=args.calibration_jsonl,
            output=args.output,
            group_size=args.group_size,
            calibration_tokens=args.calibration_tokens,
            max_samples=args.max_samples,
            max_tokens=args.max_tokens,
            seed=args.seed,
            enable_deltanet_z=args.enable_deltanet_z,
        ),
        model_adapter,
    )


def cmd_block_sensitivity(args: argparse.Namespace) -> None:
    from .quant.sensitivity import BlockSensitivityRequest, run_block_sensitivity

    model_id, model_adapter = _resolved_model(args)
    result = run_block_sensitivity(
        BlockSensitivityRequest(
            model=model_id,
            revision=args.revision,
            attn_implementation=args.attn_implementation,
            calibration_jsonl=args.calibration_jsonl,
            awq_diagnostics=args.awq_diagnostics,
            output=args.output,
            group_size=args.group_size,
            high_count=args.high_count,
            middle_count=args.middle_count,
            max_samples=args.max_samples,
            max_tokens=args.max_tokens,
            seed=args.seed,
        ),
        model_adapter,
    )
    print(json.dumps(result, sort_keys=True))


def cmd_opd_rollout(args: argparse.Namespace) -> None:
    from .opd.rollout import RolloutRequest, run_rollout

    _, model_adapter = _resolved_model(args)
    result = run_rollout(
        RolloutRequest(
            student=_student(args),
            prompts=args.prompts,
            output=args.output,
            max_tokens=args.max_tokens,
            max_new_tokens=args.max_new_tokens,
            thinking=args.thinking,
            reasoning_effort=args.reasoning_effort,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            seed=args.seed,
        ),
        model_adapter,
    )
    print(json.dumps(result))


def cmd_opd_teacher(args: argparse.Namespace) -> None:
    from .opd.teacher import TeacherRequest, run_teacher

    model_id, model_adapter = _resolved_model(args)
    result = run_teacher(
        TeacherRequest(
            model=model_id,
            revision=args.revision,
            attn_implementation=args.attn_implementation,
            rollouts=args.rollouts,
            output=args.output,
            max_tokens=args.max_tokens,
        ),
        model_adapter,
    )
    print(json.dumps(result))


def cmd_opd_shard_rollouts(args: argparse.Namespace) -> None:
    from .opd.artifacts import shard_valid_rollouts

    result = shard_valid_rollouts(
        args.rollouts,
        args.output,
        shard_size=args.shard_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_tokens=args.max_tokens,
    )
    print(json.dumps({"shards": len(result["shards"]), "samples": result["total_samples"]}))


def cmd_opd_train(args: argparse.Namespace) -> None:
    from .opd.trainer import TrainerRequest, run_training

    _, model_adapter = _resolved_model(args)
    result = run_training(
        TrainerRequest(
            student=_student(args),
            teacher_artifacts=args.teacher_artifacts,
            output=args.output,
            token_chunk_size=args.token_chunk_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            lr_scheduler_type=args.lr_scheduler_type,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            epochs=args.epochs,
            max_grad_norm=args.max_grad_norm,
            gradient_checkpointing=args.gradient_checkpointing,
            seed=args.seed,
            logging_dir=args.logging_dir,
            logging_flush_secs=args.logging_flush_secs,
            checkpoint_interval_fraction=args.checkpoint_interval_fraction,
            save_total_limit=args.save_total_limit,
            resume_from_checkpoint=args.resume_from_checkpoint,
            training_plan=args.training_plan,
            shard_index=args.shard_index,
        ),
        model_adapter,
    )
    print(json.dumps(result))


def cmd_opd_export(args: argparse.Namespace) -> None:
    from .opd.export import ReferenceExportRequest, export_reference_student

    _, model_adapter = _resolved_model(args)
    result = export_reference_student(
        ReferenceExportRequest(
            student=_student(args),
            output=args.output,
            max_shard_bytes=args.max_shard_bytes,
        ),
        model_adapter,
    )
    print(json.dumps(result))


def cmd_runtime_pack_reference(args: argparse.Namespace) -> None:
    from .runtime import pack_reference_checkpoint

    print(json.dumps(pack_reference_checkpoint(args.source, args.output)))


def cmd_runtime_verify_packed(args: argparse.Namespace) -> None:
    from .runtime import verify_packed_checkpoint

    print(json.dumps(verify_packed_checkpoint(args.packed, source=args.source)))


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--model-adapter")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--attn-implementation", default="sdpa")


def _add_student_args(parser: argparse.ArgumentParser) -> None:
    _add_model_args(parser)
    parser.add_argument("--adapter-type", choices=("dora", "lora"), default="dora")
    parser.add_argument("--mode", choices=("ste", "fast"), default="ste")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--clip-ratios")
    parser.add_argument("--adapter", help="recovery-adapter checkpoint directory")
    parser.add_argument("--adapter-rank", type=int, default=8)
    parser.add_argument("--adapter-alpha", type=float, default=16.0)
    parser.add_argument("--enable-deltanet-z", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="latticerun")
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init-config")
    init.add_argument("--output", default="latticerun.json")
    init.set_defaults(func=cmd_init)
    commands.add_parser("doctor").set_defaults(func=cmd_doctor)

    inspect = commands.add_parser("inspect")
    _add_model_args(inspect)
    inspect.add_argument("--device", default="cuda")
    inspect.add_argument("--enable-deltanet-z", action="store_true")
    inspect.set_defaults(func=cmd_inspect)

    quantize = commands.add_parser("quantize")
    _add_model_args(quantize)
    quantize.add_argument("--calibration-jsonl", required=True)
    quantize.add_argument("--output", required=True)
    quantize.add_argument("--group-size", type=int, default=128)
    quantize.add_argument("--calibration-tokens", type=int, default=512)
    quantize.add_argument("--max-samples", type=int, default=32)
    quantize.add_argument("--max-tokens", type=int, default=16_384)
    quantize.add_argument("--seed", type=int, default=42)
    quantize.add_argument("--enable-deltanet-z", action="store_true")
    quantize.set_defaults(func=cmd_quantize)

    sensitivity = commands.add_parser("sensitivity")
    sensitivity_commands = sensitivity.add_subparsers(required=True)
    block = sensitivity_commands.add_parser("block")
    _add_model_args(block)
    block.add_argument("--calibration-jsonl", required=True)
    block.add_argument("--awq-diagnostics", required=True)
    block.add_argument("--output", required=True)
    block.add_argument("--group-size", type=int, default=128)
    block.add_argument("--high-count", type=int, default=8)
    block.add_argument("--middle-count", type=int, default=2)
    block.add_argument("--max-samples", type=int, default=1)
    block.add_argument("--max-tokens", type=int, default=512)
    block.add_argument("--seed", type=int, default=42)
    block.set_defaults(func=cmd_block_sensitivity)

    opd = commands.add_parser("opd")
    opd_commands = opd.add_subparsers(required=True)
    rollout = opd_commands.add_parser("rollout")
    _add_student_args(rollout)
    rollout.add_argument("--prompts", required=True)
    rollout.add_argument("--output", required=True)
    rollout.add_argument("--max-tokens", type=int, default=16_384)
    rollout.add_argument("--max-new-tokens", type=int, default=4096)
    rollout.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=True)
    rollout.add_argument("--reasoning-effort", choices=("low", "medium", "xhigh"), default="medium")
    rollout.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=True)
    rollout.add_argument("--temperature", type=float, default=1.0)
    rollout.add_argument("--top-p", type=float, default=0.95)
    rollout.add_argument("--top-k", type=int, default=20)
    rollout.add_argument("--seed", type=int, default=42)
    rollout.set_defaults(func=cmd_opd_rollout)

    teacher = opd_commands.add_parser("teacher")
    _add_model_args(teacher)
    teacher.add_argument("--rollouts", required=True)
    teacher.add_argument("--output", required=True)
    teacher.add_argument("--max-tokens", type=int, default=16_384)
    teacher.set_defaults(func=cmd_opd_teacher)

    shard = opd_commands.add_parser("shard-rollouts")
    shard.add_argument("--rollouts", required=True)
    shard.add_argument("--output", required=True)
    shard.add_argument("--shard-size", type=int, default=480)
    shard.add_argument("--gradient-accumulation-steps", type=int, default=8)
    shard.add_argument("--max-tokens", type=int, default=16_384)
    shard.set_defaults(func=cmd_opd_shard_rollouts)

    train = opd_commands.add_parser("train")
    _add_student_args(train)
    train.add_argument("--teacher-artifacts", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--token-chunk-size", type=int, default=256)
    train.add_argument("--learning-rate", type=float, default=1e-4)
    train.add_argument("--weight-decay", type=float, default=0.0)
    train.add_argument("--warmup-ratio", type=float, default=0.03)
    train.add_argument("--lr-scheduler-type", choices=("cosine",), default="cosine")
    train.add_argument("--gradient-accumulation-steps", type=int, default=8)
    train.add_argument("--epochs", type=int, default=1)
    train.add_argument("--max-grad-norm", type=float, default=1.0)
    train.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    train.add_argument("--logging-dir")
    train.add_argument("--logging-flush-secs", type=int, default=10)
    train.add_argument("--checkpoint-interval-fraction", type=float, default=1 / 3)
    train.add_argument("--save-total-limit", type=int, default=1)
    train.add_argument("--resume-from-checkpoint")
    train.add_argument("--training-plan")
    train.add_argument("--shard-index", type=int)
    train.add_argument("--seed", type=int, default=42)
    train.set_defaults(func=cmd_opd_train)

    export = opd_commands.add_parser("export")
    _add_student_args(export)
    export.add_argument("--output", required=True)
    export.add_argument("--max-shard-bytes", type=int, default=3_500_000_000)
    export.set_defaults(func=cmd_opd_export)

    runtime = commands.add_parser("runtime")
    runtime_commands = runtime.add_subparsers(required=True)
    pack_reference = runtime_commands.add_parser("pack-reference")
    pack_reference.add_argument("--source", required=True)
    pack_reference.add_argument("--output", required=True)
    pack_reference.set_defaults(func=cmd_runtime_pack_reference)
    verify_packed = runtime_commands.add_parser("verify-packed")
    verify_packed.add_argument("--packed", required=True)
    verify_packed.add_argument("--source")
    verify_packed.set_defaults(func=cmd_runtime_verify_packed)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
