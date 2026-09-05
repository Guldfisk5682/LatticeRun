import pytest
import torch
from latticerun.model import assert_text_only_model, load_text_only_causal_lm
from safetensors.torch import save_file
from transformers.models.qwen3_5.configuration_qwen3_5 import (
    Qwen3_5Config,
    Qwen3_5TextConfig,
)
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM


def test_vision_tower_is_hard_rejected():
    model = torch.nn.Module()
    model.visual = torch.nn.Linear(2, 2)
    with pytest.raises(RuntimeError, match="vision tower"):
        assert_text_only_model(model)


def test_text_only_model_is_allowed():
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2)])
    assert_text_only_model(model)


def test_composite_checkpoint_keys_load_into_text_only_model(tmp_path):
    config = Qwen3_5TextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        layer_types=[
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ],
        max_position_embeddings=32,
        bos_token_id=0,
        eos_token_id=1,
    )
    original = Qwen3_5ForCausalLM(config)
    original.save_pretrained(tmp_path, safe_serialization=True)
    composite_keys = {
        (
            "model.language_model." + key[6:] if key.startswith("model.") else key
        ): value.detach().clone()
        for key, value in original.state_dict().items()
    }
    save_file(composite_keys, tmp_path / "model.safetensors")
    Qwen3_5Config(text_config=config).save_pretrained(tmp_path)
    loaded = load_text_only_causal_lm(
        tmp_path,
        dtype=torch.float32,
    )
    torch.testing.assert_close(
        loaded.model.embed_tokens.weight, original.model.embed_tokens.weight
    )
    torch.testing.assert_close(loaded.lm_head.weight, original.lm_head.weight)
    assert_text_only_model(loaded)
