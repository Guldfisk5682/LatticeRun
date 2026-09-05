import torch
from latticerun.quant import (
    choose_awq_clip_ratio,
    choose_weight_mse_clip_ratio,
    fake_quantize,
    quantize_symmetric_groupwise,
    ste_quantize,
)
from latticerun.quant.sensitivity import (
    normalized_block_output_error,
    select_block_sensitivity_targets,
)


def test_int3_symmetric_range_and_padding():
    torch.manual_seed(1)
    weight = torch.randn(5, 131)
    result = quantize_symmetric_groupwise(weight, bits=3, group_size=128)
    assert result.codes.min() >= -3
    assert result.codes.max() <= 3
    assert result.padded_columns == 125
    assert result.dequantize().shape == weight.shape
    assert torch.isfinite(result.dequantize()).all()


def test_zero_group_is_finite():
    result = quantize_symmetric_groupwise(torch.zeros(2, 8), bits=3, group_size=4)
    assert torch.equal(result.dequantize(), torch.zeros(2, 8, dtype=torch.float16))


def test_ste_passes_identity_gradient():
    weight = torch.tensor([[0.2, -0.3, 0.8, 1.1]], requires_grad=True)
    ste_quantize(weight, bits=3, group_size=4).sum().backward()
    torch.testing.assert_close(weight.grad, torch.ones_like(weight))


def test_export_scales_can_remain_fp32_for_runtime_repacking():
    result = quantize_symmetric_groupwise(
        torch.randn(4, 8), bits=3, group_size=4, scale_dtype=torch.float32
    )
    assert result.scales.dtype == torch.float32


def test_awq_uses_activation_output_error():
    weight = torch.tensor([[20.0, 1.0, 1.0, 1.0], [0.5, -0.5, 0.5, -0.5]])
    activations = torch.tensor([[0.0, 1.0, 1.0, 1.0]]).repeat(8, 1)
    ratio, errors = choose_awq_clip_ratio(
        weight,
        activations,
        bits=3,
        group_size=4,
        ratios=(1.0, 0.8, 0.5),
        token_batch_size=2,
    )
    assert ratio in errors
    assert errors[ratio] == min(errors.values())
    assert torch.isfinite(fake_quantize(weight)).all()


def test_weight_mse_reference_selects_minimum_grid_candidate():
    weight = torch.tensor([[9.0, 1.0, -1.0, 0.2]])
    ratio, errors = choose_weight_mse_clip_ratio(
        weight, bits=3, group_size=4, ratios=(1.0, 0.9, 0.8)
    )
    assert errors[ratio] == min(errors.values())


def test_block_error_matches_global_normalized_output_formula():
    reference = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    candidate = reference + torch.tensor([[[1.0, 0.0], [0.0, -2.0]]])
    numerator, denominator = normalized_block_output_error(reference, candidate)
    assert numerator == 5.0
    assert denominator == 30.0
    assert numerator / denominator == 1.0 / 6.0


def test_block_sensitivity_selects_highest_and_global_middle_targets():
    diagnostics = {
        f"model.layers.{index}.mlp.up_proj": {
            "awq_local_normalized_error": float(index),
            "awq_clip_ratio": 1.0 - index / 100,
        }
        for index in range(12)
    }
    selected = select_block_sensitivity_targets(
        diagnostics,
        high_count=3,
        middle_count=2,
    )
    assert [target.local_error for target in selected[:3]] == [11.0, 10.0, 9.0]
    assert [target.local_error for target in selected[3:]] == [5.0, 6.0]
    assert [target.selection for target in selected] == [
        "highest",
        "highest",
        "highest",
        "middle",
        "middle",
    ]
