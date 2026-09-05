import pytest
from latticerun.model import QWEN35_ADAPTER, QwenRole, classify_module, decide_module


@pytest.mark.parametrize(
    ("name", "role"),
    [
        ("model.embed_tokens", QwenRole.EMBEDDING),
        ("lm_head", QwenRole.LM_HEAD),
        ("model.layers.3.mlp.gate_proj", QwenRole.FFN),
        ("model.layers.3.self_attn.q_proj", QwenRole.FULL_ATTENTION),
        ("model.layers.2.linear_attn.in_proj_qkv", QwenRole.DELTANET_QKV),
        ("model.layers.2.linear_attn.in_proj_z", QwenRole.DELTANET_Z),
        ("model.layers.2.linear_attn.in_proj_a", QwenRole.DELTANET_CONTROL),
        ("model.layers.2.linear_attn.in_proj_b", QwenRole.DELTANET_CONTROL),
        ("model.layers.2.linear_attn.out_proj", QwenRole.DELTANET_OUT),
        ("model.layers.2.linear_attn.conv1d", QwenRole.CONV),
    ],
)
def test_qwen38_module_classification(name, role):
    assert classify_module(name, 2_000_000) == role


def test_qwen38_adapter_resolves_enclosing_decoder_block():
    assert (
        QWEN35_ADAPTER.block_name_for_module("model.layers.25.linear_attn.out_proj")
        == "model.layers.25"
    )
    with pytest.raises(ValueError, match="not inside a decoder block"):
        QWEN35_ADAPTER.block_name_for_module("lm_head")


def test_phase_one_precision_and_opd_policy():
    assert decide_module("model.layers.0.mlp.down_proj", 10_000_000).bits == 3
    assert decide_module("model.layers.0.mlp.down_proj", 10_000_000).opd_eligible
    assert decide_module("lm_head", 10_000_000).bits == 4
    a = decide_module("model.layers.0.linear_attn.in_proj_a", 10_000_000)
    assert a.bits is None
    assert a.compute_dtype == "bf16"
    z = decide_module("model.layers.0.linear_attn.in_proj_z", 10_000_000)
    assert z.bits == 3
    assert not z.opd_eligible
    enabled_z = decide_module(
        "model.layers.0.linear_attn.in_proj_z", 10_000_000, enable_deltanet_z=True
    )
    assert enabled_z.bits == 3
    assert enabled_z.opd_eligible
