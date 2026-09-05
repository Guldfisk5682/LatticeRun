import pytest
import torch
from latticerun.adapters import (
    FrozenQuantLinear,
    QuantLoRALinear,
    merge_recovery_for_inference,
)
from latticerun.config import RecoveryConfig
from latticerun.quant import fake_quantize


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA regression")
def test_fast_merge_requantize_stays_on_execution_device():
    source = torch.nn.Linear(8, 6, bias=False, device="cuda", dtype=torch.float16)
    layer = QuantLoRALinear(
        source,
        RecoveryConfig(adapter_type="lora", rank=2, alpha=4, mode="fast"),
        bits=3,
        group_size=4,
        clip_ratio=1.0,
    )
    assert layer.base_weight.device.type == "cpu"
    layer.lora_b.data.normal_()
    model = torch.nn.Sequential(layer)
    merge_recovery_for_inference(model)
    assert model[0].weight.device.type == "cuda"
    output = model(torch.randn(2, 8, device="cuda", dtype=torch.float16))
    assert output.device.type == "cuda"


def test_merge_requantizes_the_effective_weight_on_cpu():
    torch.manual_seed(3)
    source = torch.nn.Linear(8, 6, bias=True, dtype=torch.bfloat16)
    layer = QuantLoRALinear(
        source,
        RecoveryConfig(adapter_type="lora", rank=2, alpha=4, mode="ste"),
        bits=3,
        group_size=4,
        clip_ratio=0.9,
    )
    layer.lora_a.data.normal_(std=0.1)
    layer.lora_b.data.normal_(std=0.1)
    expected = fake_quantize(
        layer.effective_weight(), bits=3, group_size=4, clip_ratio=0.9
    )
    model = torch.nn.Sequential(layer)

    merge_recovery_for_inference(model)

    assert isinstance(model[0], FrozenQuantLinear)
    torch.testing.assert_close(model[0].weight, expected)
