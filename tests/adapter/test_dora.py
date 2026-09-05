import torch
import torch.nn.functional as F
from latticerun.adapters import QuantDoRALinear, merge_recovery_for_inference
from latticerun.config import RecoveryConfig
from latticerun.quant import fake_quantize
from peft import LoraConfig as PeftLoraConfig
from peft import get_peft_model


class _TinyModel(torch.nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.linear = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False)
        self.linear.weight.data.copy_(weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs)


def _custom_and_peft():
    torch.manual_seed(21)
    weight = torch.randn(6, 8)
    source = torch.nn.Linear(8, 6, bias=False)
    source.weight.data.copy_(weight)
    custom = QuantDoRALinear(
        source,
        RecoveryConfig(adapter_type="dora", rank=2, alpha=4, mode="ste"),
        bits=3,
        group_size=4,
        clip_ratio=1.0,
    )
    peft_model = get_peft_model(
        _TinyModel(weight),
        PeftLoraConfig(
            r=2,
            lora_alpha=4,
            target_modules=["linear"],
            use_dora=True,
            bias="none",
        ),
    )
    peft_layer = peft_model.base_model.model.linear
    custom.lora_a.data.normal_(mean=0.0, std=0.2)
    custom.lora_b.data.normal_(mean=0.0, std=0.2)
    custom.magnitude.data.mul_(torch.linspace(0.8, 1.2, 6))
    peft_layer.lora_A["default"].weight.data.copy_(custom.lora_a)
    peft_layer.lora_B["default"].weight.data.copy_(custom.lora_b)
    peft_layer.lora_magnitude_vector["default"].weight.data.copy_(custom.magnitude)
    return custom, peft_model, peft_layer


def test_dora_effective_weight_and_gradients_match_peft_reference():
    custom, peft_model, peft_layer = _custom_and_peft()
    inputs = torch.randn(4, 8)
    custom_output = F.linear(inputs, custom.effective_weight())
    peft_output = peft_model(inputs)
    torch.testing.assert_close(custom_output, peft_output, rtol=2e-5, atol=2e-6)

    custom_output.sum().backward()
    peft_output.sum().backward()
    torch.testing.assert_close(
        custom.lora_a.grad, peft_layer.lora_A["default"].weight.grad
    )
    torch.testing.assert_close(
        custom.lora_b.grad, peft_layer.lora_B["default"].weight.grad
    )
    torch.testing.assert_close(
        custom.magnitude.grad,
        peft_layer.lora_magnitude_vector["default"].weight.grad,
    )


def test_dora_ste_forward_and_merge_use_same_effective_weight():
    custom, _, _ = _custom_and_peft()
    inputs = torch.randn(3, 8)
    expected_weight = fake_quantize(custom.effective_weight(), bits=3, group_size=4)
    torch.testing.assert_close(custom(inputs), F.linear(inputs, expected_weight))

    model = torch.nn.Sequential(custom)
    merge_recovery_for_inference(model)
    torch.testing.assert_close(model(inputs), F.linear(inputs, expected_weight))
