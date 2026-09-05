import json
from dataclasses import asdict

import pytest
import torch
from latticerun.adapters import QuantLoRALinear, trainable_adapter_parameters
from latticerun.config import RecoveryConfig
from latticerun.opd import (
    RolloutRecord,
    backward_chunked_forward_kl,
    chunked_forward_kl_from_hidden,
    dense_forward_kl_from_hidden,
    read_valid_rollouts,
)
from latticerun.opd.artifacts import load_rollout_shard_plan, shard_valid_rollouts
from latticerun.opd.checkpoint import (
    TrainingProgress,
    _prune_checkpoints,
    artifact_fingerprint,
    checkpoint_interval_steps,
    load_training_checkpoint,
    save_training_checkpoint,
)
from latticerun.opd.trainer import _artifact_completion_tokens, _warmup_steps
from safetensors.torch import save_file


def _inputs(requires_grad=True):
    torch.manual_seed(8)
    student = torch.randn(2, 7, 5, requires_grad=requires_grad)
    teacher = torch.randn(2, 7, 5)
    student_head = torch.randn(11, 5)
    teacher_head = torch.randn(11, 5)
    mask = torch.tensor(
        [
            [False, False, True, True, True, True, True],
            [False, False, False, True, True, True, False],
        ]
    )
    return student, teacher, student_head, teacher_head, mask


def test_token_chunked_kl_equals_dense_full_vocab_kl():
    values = _inputs()
    dense = dense_forward_kl_from_hidden(*values)
    chunked = chunked_forward_kl_from_hidden(*values, token_chunk_size=3)
    torch.testing.assert_close(chunked, dense, rtol=1e-6, atol=1e-6)


def test_chunked_backward_matches_dense_gradient():
    student, teacher, student_head, teacher_head, mask = _inputs()
    dense = dense_forward_kl_from_hidden(
        student, teacher, student_head, teacher_head, mask
    )
    dense.backward()
    dense_gradient = student.grad.clone()

    student2 = student.detach().clone().requires_grad_(True)
    observed = backward_chunked_forward_kl(
        student2, teacher, student_head, teacher_head, mask, token_chunk_size=2
    )
    torch.testing.assert_close(
        torch.tensor(observed), dense.detach(), rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(student2.grad, dense_gradient, rtol=1e-5, atol=1e-6)


def test_unequal_trajectories_accumulate_as_one_global_token_mean():
    torch.manual_seed(19)
    student_head = torch.randn(13, 5)
    teacher_head = torch.randn(13, 5)
    student_a = torch.randn(1, 7, 5, requires_grad=True)
    teacher_a = torch.randn(1, 7, 5)
    mask_a = torch.tensor([[False, True, True, True, True, True, True]])
    student_b = torch.randn(1, 4, 5, requires_grad=True)
    teacher_b = torch.randn(1, 4, 5)
    mask_b = torch.tensor([[False, False, True, True]])
    tokens_a = int(mask_a[:, 1:].sum())
    tokens_b = int(mask_b[:, 1:].sum())
    total_tokens = tokens_a + tokens_b

    dense_a = dense_forward_kl_from_hidden(
        student_a, teacher_a, student_head, teacher_head, mask_a
    )
    dense_b = dense_forward_kl_from_hidden(
        student_b, teacher_b, student_head, teacher_head, mask_b
    )
    dense_global = (dense_a * tokens_a + dense_b * tokens_b) / total_tokens
    dense_global.backward()
    expected_a_gradient = student_a.grad.clone()
    expected_b_gradient = student_b.grad.clone()

    observed_a = student_a.detach().clone().requires_grad_(True)
    observed_b = student_b.detach().clone().requires_grad_(True)
    mean_a = backward_chunked_forward_kl(
        observed_a,
        teacher_a,
        student_head,
        teacher_head,
        mask_a,
        token_chunk_size=2,
        gradient_denominator=total_tokens,
    )
    mean_b = backward_chunked_forward_kl(
        observed_b,
        teacher_b,
        student_head,
        teacher_head,
        mask_b,
        token_chunk_size=2,
        gradient_denominator=total_tokens,
    )
    observed_global = (mean_a * tokens_a + mean_b * tokens_b) / total_tokens

    torch.testing.assert_close(
        torch.tensor(observed_global), dense_global.detach(), rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(
        observed_a.grad, expected_a_gradient, rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(
        observed_b.grad, expected_b_gradient, rtol=1e-5, atol=1e-6
    )


def test_truncated_rollout_is_never_accepted(tmp_path):
    path = tmp_path / "rollouts.jsonl"
    record = RolloutRecord("bad", [1, 2, 3], 1, eos_reached=False, truncated=True)
    path.write_text(json.dumps(asdict(record)) + "\n")
    with pytest.raises(ValueError, match="truncated"):
        read_valid_rollouts(path)


def test_rollout_shards_preserve_order_and_accumulation_boundaries(tmp_path):
    source = tmp_path / "rollouts.jsonl"
    records = [
        RolloutRecord(
            f"sample-{index}",
            [1, 2, 3],
            1,
            eos_reached=True,
            truncated=False,
        )
        for index in range(10)
    ]
    source.write_text("".join(json.dumps(asdict(row)) + "\n" for row in records))
    output = tmp_path / "shards"
    plan = shard_valid_rollouts(
        source, output, shard_size=4, gradient_accumulation_steps=2
    )
    assert [shard["samples"] for shard in plan["shards"]] == [4, 4, 2]
    assert [
        record.sample_id
        for shard in plan["shards"]
        for record in read_valid_rollouts(output / shard["file"])
    ] == [record.sample_id for record in records]
    restored, fingerprint = load_rollout_shard_plan(output / "plan.json")
    assert restored == plan
    assert len(fingerprint) == 64
    with pytest.raises(ValueError, match="divisible"):
        shard_valid_rollouts(
            source,
            tmp_path / "invalid",
            shard_size=3,
            gradient_accumulation_steps=2,
        )


def test_teacher_artifact_token_count_matches_causal_completion_mask(tmp_path):
    path = tmp_path / "artifact.safetensors"
    save_file(
        {
            "completion_mask": torch.tensor(
                [False, False, True, True, True], dtype=torch.bool
            )
        },
        path,
    )
    assert _artifact_completion_tokens(path) == 3


def test_three_percent_warmup_rounds_up_to_a_real_optimizer_step():
    assert _warmup_steps(100, 0.03) == 3
    assert _warmup_steps(33, 0.03) == 1


def test_checkpoint_fraction_maps_sixty_three_steps_to_thirds():
    assert checkpoint_interval_steps(63, 1 / 3) == 21
    assert checkpoint_interval_steps(2, 1 / 3) == 1
    with pytest.raises(ValueError, match="fraction"):
        checkpoint_interval_steps(63, 0)


def test_checkpoint_fingerprint_and_single_retention_are_deterministic(tmp_path):
    artifacts = [tmp_path / "b.safetensors", tmp_path / "a.safetensors"]
    artifacts[0].write_bytes(b"two")
    artifacts[1].write_bytes(b"one")
    assert artifact_fingerprint(artifacts) == artifact_fingerprint(artifacts[::-1])
    for step in (21, 42, 63):
        (tmp_path / f"checkpoint-{step}").mkdir()
    (tmp_path / ".checkpoint-63-incomplete").mkdir()
    _prune_checkpoints(tmp_path, 1)
    assert sorted(path.name for path in tmp_path.iterdir() if path.is_dir()) == [
        ".checkpoint-63-incomplete",
        "checkpoint-63",
    ]


def test_checkpoint_restores_adapter_optimizer_scheduler_and_progress(tmp_path):
    recovery = RecoveryConfig(adapter_type="lora", rank=2, alpha=4, mode="ste")
    layer = QuantLoRALinear(
        torch.nn.Linear(4, 3, bias=False),
        recovery,
        bits=3,
        group_size=2,
        clip_ratio=1.0,
    )
    model = torch.nn.Sequential(layer)
    parameters = trainable_adapter_parameters(model)
    optimizer = torch.optim.AdamW(parameters, lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 0.9**step)
    model(torch.randn(2, 4)).sum().backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    expected_lora_b = layer.lora_b.detach().clone()
    progress = TrainingProgress(global_step=21, micro_step=168, next_group_start=168)
    contract = {"fingerprint": "fixed"}
    checkpoint, _ = save_training_checkpoint(
        model=model,
        recovery=recovery,
        optimizer=optimizer,
        scheduler=scheduler,
        output=tmp_path,
        progress=progress,
        telemetry_state={
            "kl_ema": 0.2,
            "sequence_tokens_seen": 200,
            "completion_tokens_seen": 180,
        },
        contract=contract,
        save_total_limit=1,
    )
    layer.lora_b.data.zero_()
    restored, telemetry = load_training_checkpoint(
        checkpoint=checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        expected_contract=contract,
        device=torch.device("cpu"),
    )
    torch.testing.assert_close(layer.lora_b, expected_lora_b)
    assert restored == progress
    assert telemetry["completion_tokens_seen"] == 180
    assert scheduler.last_epoch == 1
