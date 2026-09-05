"""Analytical packed-weight/KV/offload budget; these are not measured speeds."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

GIB = 2**30


def estimate(args: argparse.Namespace) -> dict:
    manifest = json.loads(args.manifest.read_text())
    config = manifest["text_config"]
    roles = manifest["roles"]
    group = manifest["group_size"]

    def weight_bytes(role):
        return role["groups"] * (group * role["bits"] // 8 + args.scale_bytes)

    weights = sum(weight_bytes(r) for r in roles.values())
    weights += manifest["protected_bytes_estimate"]
    embedding = weight_bytes(roles["embedding"])
    gpu_weight_target = weights - (embedding if args.cpu_embedding else 0)
    n_full = config["layer_types"].count("full_attention")
    n_gdn = config["layer_types"].count("linear_attention")
    dim = config["head_dim"]
    # Fork INT8 uses one FP32 scale per token/head for each of K and V.
    bytes_per_kv_head = {
        "bf16": 2 * dim * 2,
        "int8": 2 * (dim + 4),
        "tq_k8v4": dim + math.ceil(dim * 4 / 8) + 4,
        "tq_k4v4": math.ceil(dim * 4 / 8) * 2 + 2 + 4,
    }[args.kv_format]
    kv = n_full * args.context * config["num_key_value_heads"] * bytes_per_kv_head
    recurrent = (
        n_gdn
        * config["linear_num_value_heads"]
        * config["linear_key_head_dim"]
        * config["linear_value_head_dim"]
        * 4
    )
    conv = (
        n_gdn
        * (config["linear_conv_kernel_dim"] - 1)
        * 2
        * (
            2 * config["linear_num_key_heads"] * config["linear_key_head_dim"]
            + config["linear_num_value_heads"] * config["linear_value_head_dim"]
        )
    )
    state = (kv + recurrent + conv) * args.sequences
    layer = manifest["largest_quantized_decoder_layer"]
    # Slots hold packed quantized matrices only; protected tensors stay resident.
    slot_bytes = layer["code_bytes"] + layer["groups"] * args.scale_bytes
    fixed = state + (args.workspace_gib + args.safety_gib) * GIB
    offload = max(0, gpu_weight_target + fixed - args.vram_gib * GIB)
    buffers = args.weight_slots * slot_bytes if offload else 0
    if offload:
        offload += buffers
    feasible = offload <= gpu_weight_target - manifest["protected_bytes_estimate"]
    gpu_resident_weights = max(0, gpu_weight_target - offload)
    transfer_s = offload / (args.h2d_gbps * 1e9)
    disk_s = offload / (args.disk_gbps * 1e9)
    return {
        "kind": "analytical byte budget and optimistic transfer ceilings, not benchmark",
        "model_revision": manifest["source_revision"],
        "parameters": manifest["total_parameters"],
        "context_tokens_per_sequence": args.context,
        "sequences": args.sequences,
        "scale_bytes": args.scale_bytes,
        "kv_format": args.kv_format,
        "cpu_embedding": args.cpu_embedding,
        "packed_weights_gib": weights / GIB,
        "embedding_gib": embedding / GIB,
        "full_attention_kv_gib": kv * args.sequences / GIB,
        "fp32_recurrent_gib": recurrent * args.sequences / GIB,
        "conv_state_gib": conv * args.sequences / GIB,
        "workspace_gib_assumption": args.workspace_gib,
        "safety_gib_assumption": args.safety_gib,
        "one_packed_decoder_slot_gib": slot_bytes / GIB,
        "streaming_gpu_buffers_gib": buffers / GIB,
        "minimum_offload_gib_approx": offload / GIB,
        "gpu_resident_weights_gib": gpu_resident_weights / GIB,
        "gpu_total_gib_approx": (gpu_resident_weights + fixed + buffers) / GIB,
        "capacity_feasible_under_assumptions": feasible,
        "offloaded_payload_h2d_seconds_per_token_floor": transfer_s,
        "h2d_only_tokens_per_second_ceiling": 1 / transfer_s if transfer_s else None,
        "cold_offloaded_payload_disk_seconds_per_token_floor": disk_s,
        "disk_only_tokens_per_second_ceiling": 1 / disk_s if disk_s else None,
        "notes": [
            "Continuous byte allocation ignores layer granularity, allocator padding, and graph/state snapshots.",
            "Workspace and safety are user assumptions; reserve real peak prefill/continuation buffers before KV.",
            "Host RSS/page cache/pinned pools/CPU activations require a separate RAM budget.",
            "GPU-resident weights need no per-token disk or H2D reload; embedding is a row lookup.",
            "No compute/driver/IOPS/contention costs included; bandwidth inputs use decimal GB/s.",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--context", type=int, default=131072)
    parser.add_argument("--sequences", type=int, default=1)
    parser.add_argument("--scale-bytes", type=int, choices=(2, 4), default=4)
    parser.add_argument(
        "--kv-format", choices=("bf16", "int8", "tq_k8v4", "tq_k4v4"), default="int8"
    )
    parser.add_argument("--cpu-embedding", action="store_true")
    parser.add_argument("--vram-gib", type=float, default=16)
    parser.add_argument("--weight-slots", type=int, default=2)
    parser.add_argument("--workspace-gib", type=float, default=1)
    parser.add_argument("--safety-gib", type=float, default=1)
    parser.add_argument("--h2d-gbps", type=float, default=24)
    parser.add_argument("--disk-gbps", type=float, default=3.5)
    args = parser.parse_args()
    if (
        min(
            args.context,
            args.sequences,
            args.vram_gib,
            args.h2d_gbps,
            args.disk_gbps,
            args.weight_slots,
        )
        <= 0
    ):
        parser.error(
            "context, sequences, capacities, bandwidths and slots must be positive"
        )
    if min(args.workspace_gib, args.safety_gib) < 0:
        parser.error("workspace and safety must be nonnegative")
    print(json.dumps(estimate(args), indent=2))
