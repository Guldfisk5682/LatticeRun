import json

import torch
import torch.nn.functional as F
from latticerun.adapters import QuantLoRALinear, load_adapter, save_adapter
from latticerun.config import LoRAConfig
from latticerun.quant import fake_quantize


def _source():
    torch.manual_seed(4)
    return torch.nn.Linear(8, 6, bias=False)


def test_fast_is_parallel_quantized_base_plus_lora():
    layer = QuantLoRALinear(
        _source(),
        LoRAConfig(rank=2, alpha=4, mode="fast"),
        bits=3,
        group_size=4,
        clip_ratio=1.0,
    )
    layer.lora_b.data.normal_()
    inputs = torch.randn(3, 8)
    expected = F.linear(inputs, fake_quantize(layer.base_weight, bits=3, group_size=4))
    expected += F.linear(F.linear(inputs, layer.lora_a), layer.lora_b) * layer.scaling
    torch.testing.assert_close(layer(inputs), expected)


def test_ste_forward_matches_merged_then_quantized_weight():
    layer = QuantLoRALinear(
        _source(),
        LoRAConfig(rank=2, alpha=4, mode="ste"),
        bits=3,
        group_size=4,
        clip_ratio=0.9,
    )
    layer.lora_b.data.normal_()
    inputs = torch.randn(3, 8)
    merged = layer.base_weight + layer.delta_weight()
    expected = F.linear(
        inputs, fake_quantize(merged, bits=3, group_size=4, clip_ratio=0.9)
    )
    torch.testing.assert_close(layer(inputs), expected)
    layer(inputs).sum().backward()
    assert layer.lora_a.grad is not None
    assert layer.lora_b.grad is not None


def test_legacy_lora_metadata_remains_loadable(tmp_path):
    config = LoRAConfig(rank=2, alpha=4, mode="ste")
    source = QuantLoRALinear(_source(), config, bits=3, group_size=4, clip_ratio=1.0)
    source.lora_b.data.normal_()
    save_adapter(torch.nn.Sequential(source), tmp_path, config)

    metadata_path = tmp_path / "adapter_config.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert "dropout" not in metadata["recovery"]
    metadata["lora"] = metadata.pop("recovery")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    target = QuantLoRALinear(_source(), config, bits=3, group_size=4, clip_ratio=1.0)
    load_adapter(torch.nn.Sequential(target), tmp_path)
    torch.testing.assert_close(target.lora_a, source.lora_a)
    torch.testing.assert_close(target.lora_b, source.lora_b)
