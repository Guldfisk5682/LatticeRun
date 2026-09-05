"""Text-only Qwen3.8 loading and model-structure checks."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from enum import Enum

import torch

from .base import ModuleDecision, ReasoningEffort, assert_text_only_model

QWEN38_MODEL_ID = "Qwen/Qwen3.8-27B"


class QwenRole(str, Enum):
    EMBEDDING = "embedding"
    LM_HEAD = "lm_head"
    FFN = "ffn"
    FULL_ATTENTION = "full_attention"
    DELTANET_QKV = "deltanet_qkv"
    DELTANET_Z = "deltanet_z"
    DELTANET_CONTROL = "deltanet_control"
    DELTANET_OUT = "deltanet_out"
    NORM = "norm"
    CONV = "conv"
    VISION = "vision"
    OTHER_SMALL = "other_small"
    UNKNOWN_LARGE = "unknown_large"

_FFN_LEAVES = {"gate_proj", "up_proj", "down_proj"}
_FULL_ATTN_LEAVES = {"q_proj", "k_proj", "v_proj", "o_proj"}


class Qwen35Adapter:
    """Qwen3.5/Qwen3.8 hybrid-text precision and recovery policy."""

    name = "qwen35"

    def load_model(
        self,
        model_id: str,
        *,
        revision: str,
        dtype: torch.dtype,
        device_map: str | dict[str, object] | None,
        attn_implementation: str | None,
    ) -> torch.nn.Module:
        return load_text_only_causal_lm(
            model_id,
            revision=revision,
            dtype=dtype,
            device_map=device_map,
            attn_implementation=attn_implementation,
        )

    @staticmethod
    def render_prompt(
        tokenizer: object,
        row: dict[str, object],
        *,
        thinking: bool,
        reasoning_effort: ReasoningEffort | None,
        force_chat_template: bool,
    ) -> str:
        template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": thinking,
        }
        if reasoning_effort is not None:
            template_kwargs["reasoning_effort"] = reasoning_effort
        messages = row.get("messages")
        if isinstance(messages, list):
            return tokenizer.apply_chat_template(messages, **template_kwargs)
        prompt = row.get("prompt")
        if not isinstance(prompt, str):
            raise TypeError("each row must contain `prompt` or `messages`")
        if not force_chat_template:
            return prompt
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            **template_kwargs,
        )

    @staticmethod
    def block_name_for_module(module_name: str) -> str:
        parts = module_name.split(".")
        try:
            layers_index = parts.index("layers")
        except ValueError as error:
            raise ValueError(
                f"module is not inside a decoder block: {module_name}"
            ) from error
        block_index = layers_index + 1
        if block_index >= len(parts) or not parts[block_index].isdigit():
            raise ValueError(
                f"module has no numeric decoder-block index: {module_name}"
            )
        if block_index + 1 >= len(parts):
            raise ValueError(f"module is the decoder block, not a leaf: {module_name}")
        return ".".join(parts[: block_index + 1])

    @staticmethod
    def classify_module(name: str, numel: int = 0) -> QwenRole:
        lowered = name.lower()
        leaf = lowered.rsplit(".", 1)[-1]
        if "visual" in lowered or "vision" in lowered:
            return QwenRole.VISION
        if leaf in {"embed_tokens", "embedding"}:
            return QwenRole.EMBEDDING
        if leaf == "lm_head":
            return QwenRole.LM_HEAD
        if "norm" in leaf or "layernorm" in lowered:
            return QwenRole.NORM
        if "conv1d" in lowered or leaf.startswith("conv"):
            return QwenRole.CONV
        if ".mlp." in lowered and leaf in _FFN_LEAVES:
            return QwenRole.FFN
        if ".self_attn." in lowered and leaf in _FULL_ATTN_LEAVES:
            return QwenRole.FULL_ATTENTION
        if ".linear_attn." in lowered:
            if leaf == "in_proj_qkv":
                return QwenRole.DELTANET_QKV
            if leaf == "in_proj_z":
                return QwenRole.DELTANET_Z
            if leaf in {"in_proj_a", "in_proj_b", "a_log", "dt_bias"}:
                return QwenRole.DELTANET_CONTROL
            if leaf == "out_proj":
                return QwenRole.DELTANET_OUT
        return QwenRole.UNKNOWN_LARGE if numel >= 1_000_000 else QwenRole.OTHER_SMALL

    def decide_module(
        self,
        name: str,
        numel: int = 0,
        *,
        enable_deltanet_z: bool = False,
    ) -> ModuleDecision:
        role = self.classify_module(name, numel)
        if role == QwenRole.LM_HEAD:
            return ModuleDecision(name, role, 4, "bf16", False, "lm_head uses INT4")
        if role in {
            QwenRole.EMBEDDING,
            QwenRole.FFN,
            QwenRole.FULL_ATTENTION,
            QwenRole.DELTANET_QKV,
            QwenRole.DELTANET_OUT,
        }:
            opd = role in {
                QwenRole.FFN,
                QwenRole.FULL_ATTENTION,
                QwenRole.DELTANET_QKV,
                QwenRole.DELTANET_OUT,
            }
            return ModuleDecision(
                name, role, 3, "bf16", opd, "phase-one symmetric INT3"
            )
        if role == QwenRole.DELTANET_Z:
            return ModuleDecision(
                name,
                role,
                3,
                "bf16",
                enable_deltanet_z,
                "INT3 by default; OPD is gated by sensitivity",
            )
        if role == QwenRole.DELTANET_CONTROL:
            return ModuleDecision(
                name,
                role,
                None,
                "bf16",
                False,
                "in_proj_a/b stay unquantized; g arithmetic and A_log/dt_bias are FP32",
            )
        if role in {
            QwenRole.NORM,
            QwenRole.CONV,
            QwenRole.OTHER_SMALL,
            QwenRole.VISION,
        }:
            dtype = "fp32" if role == QwenRole.NORM else "bf16"
            return ModuleDecision(
                name, role, None, dtype, False, "protected phase-one module"
            )
        return ModuleDecision(
            name,
            role,
            None,
            "bf16",
            False,
            "unclassified large tensor: inspection must fail",
        )

    def inspect_named_modules(
        self,
        named_modules: Iterable[tuple[str, object]],
        *,
        enable_deltanet_z: bool = False,
        strict: bool = True,
    ) -> list[ModuleDecision]:
        decisions: list[ModuleDecision] = []
        for name, module in named_modules:
            weight = getattr(module, "weight", None)
            if weight is None:
                continue
            decision = self.decide_module(
                name,
                int(weight.numel()),
                enable_deltanet_z=enable_deltanet_z,
            )
            decisions.append(decision)
        unknown = [
            item.name for item in decisions if item.role == QwenRole.UNKNOWN_LARGE
        ]
        if strict and unknown:
            preview = ", ".join(unknown[:8])
            raise RuntimeError(
                f"Refusing to continue: unclassified large modules: {preview}"
            )
        return decisions

    @staticmethod
    def forward_hidden(
        model: torch.nn.Module, **model_inputs: torch.Tensor
    ) -> torch.Tensor:
        outputs = model.model(**model_inputs, return_dict=True)
        return outputs.last_hidden_state


QWEN35_ADAPTER = Qwen35Adapter()


def classify_module(name: str, numel: int = 0) -> QwenRole:
    return QWEN35_ADAPTER.classify_module(name, numel)


def decide_module(
    name: str, numel: int = 0, *, enable_deltanet_z: bool = False
) -> ModuleDecision:
    return QWEN35_ADAPTER.decide_module(
        name, numel, enable_deltanet_z=enable_deltanet_z
    )


def inspect_named_modules(
    named_modules: Iterable[tuple[str, object]],
    *,
    enable_deltanet_z: bool = False,
    strict: bool = True,
) -> list[ModuleDecision]:
    return QWEN35_ADAPTER.inspect_named_modules(
        named_modules, enable_deltanet_z=enable_deltanet_z, strict=strict
    )


def load_text_config(
    model_id: str = QWEN38_MODEL_ID, *, revision: str = "main"
) -> object:
    from transformers import AutoConfig

    composite = AutoConfig.from_pretrained(model_id, revision=revision)
    text_config = getattr(composite, "text_config", None)
    if text_config is None:
        raise RuntimeError("Qwen3.8 composite config has no text_config")
    validate_qwen38_text_config(text_config)
    return text_config


def validate_qwen38_text_config(config: object) -> None:
    layer_types = list(getattr(config, "layer_types", []))
    if not layer_types:
        raise RuntimeError("Qwen3.8 text config has no layer_types")
    unexpected = set(layer_types) - {"linear_attention", "full_attention"}
    if unexpected:
        raise RuntimeError(f"unexpected Qwen3.8 layer types: {sorted(unexpected)}")
    if "linear_attention" not in layer_types or "full_attention" not in layer_types:
        raise RuntimeError(
            "Qwen3.8 must contain both DeltaNet and Full Attention layers"
        )
    interval = int(getattr(config, "full_attention_interval", 4))
    for index, layer_type in enumerate(layer_types):
        expected = (
            "full_attention" if (index + 1) % interval == 0 else "linear_attention"
        )
        if layer_type != expected:
            raise RuntimeError(
                f"unexpected hybrid schedule at layer {index}: expected {expected}, got {layer_type}"
            )


def load_text_only_causal_lm(
    model_id: str = QWEN38_MODEL_ID,
    *,
    revision: str = "main",
    dtype: torch.dtype = torch.bfloat16,
    device_map: str | dict[str, object] | None = None,
    attn_implementation: str | None = None,
) -> torch.nn.Module:
    """Load the language model without ever constructing the vision tower.

    The official checkpoint is composite and stores text weights below
    ``model.language_model``. Transformers' key mapping remaps those weights
    into ``Qwen3_5ForCausalLM.model`` while ignoring vision keys.
    """

    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    composite = AutoConfig.from_pretrained(model_id, revision=revision)
    text_config = composite.text_config
    validate_qwen38_text_config(text_config)
    kwargs: dict[str, object] = {
        "config": text_config,
        "revision": revision,
        "dtype": dtype,
        "device_map": device_map,
        "key_mapping": {r"^model\.language_model\.": "model."},
    }
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation
    model = Qwen3_5ForCausalLM.from_pretrained(model_id, **kwargs)
    assert_text_only_model(model)
    force_deltanet_control_fp32(model)
    return model


def force_deltanet_control_fp32(model: torch.nn.Module) -> None:
    """Keep g arithmetic parameters in FP32 without breaking Linear dtypes.

    Qwen3.8's upstream forward explicitly promotes ``in_proj_a(x)`` to FP32
    before computing g. The in_proj_a/b weights remain unquantized BF16 so
    their Linear calls continue to match BF16 hidden states.
    """

    for name, parameter in model.named_parameters():
        lowered = name.lower()
        if ".linear_attn." not in lowered:
            continue
        if lowered.endswith(("a_log", "dt_bias")):
            parameter.data = parameter.data.float()


def iter_text_linears(model: torch.nn.Module) -> Iterator[tuple[str, torch.nn.Linear]]:
    assert_text_only_model(model)
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            yield name, module
