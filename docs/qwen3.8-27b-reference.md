# Qwen3.8-27B Dense Reference

This example applies activation-aware mixed INT3/INT4 quantization and one
epoch of DoRA+STE on-policy distillation to the text-only Qwen3.8-27B model.

## Results

All results use greedy pass@1 with thinking enabled and
`reasoning_effort=medium`.

| Variant | MATH-500 | GPQA Diamond | HumanEval | HumanEval+ |
|---|---:|---:|---:|---:|
| Q3 | 398/500 (79.60%) | 129/198 (65.15%) | 72/164 (43.90%) | 70/164 (42.68%) |
| OPD | 444/500 (88.80%) | 150/198 (75.76%) | 133/164 (81.10%) | 128/164 (78.05%) |
| BF16 | 415/500 (83.00%) | 166/198 (83.84%) | 161/164 (98.17%) | 153/164 (93.29%) |

The Q3 variant uses symmetric groupwise INT3-G128 for the token embedding, FFN
projections, Full Attention projections, DeltaNet `in_proj_qkv`, DeltaNet
`in_proj_z`, and DeltaNet `out_proj`. The `lm_head` uses INT4-G128. DeltaNet
`in_proj_a` and `in_proj_b`, convolution, normalization, and small control
parameters remain unquantized; DeltaNet gate/control arithmetic uses FP32.

Each quantized module selects its clipping ratio by activation-weighted output
MSE over the following grid:

```text
1.00, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55
```

The OPD variant adds DoRA to 352 INT3 modules: 192 FFN projections, 64 Full
Attention projections, 48 DeltaNet `in_proj_qkv` projections, and 48 DeltaNet
`out_proj` projections. DeltaNet `in_proj_z` is quantized but is not an OPD
target. After training, the DoRA parameters are merged and the effective
weights are requantized with the selected per-module clipping ratios.

## Training data

The training mixture contains 5,000 prompt-only rows sampled with seed 42.
Assistant answers are removed before rollout, and the combined rows are
shuffled deterministically.

| Source | Rows | Share | Stratification |
|---|---:|---:|---|
| `AryaYT/nl2shell-terminal-bench` | 500 | 10% | Selected from the training pool. |
| `CodeResearch/Code-Evol-Instruct-OSS` | 2,000 | 40% | Complexity 0/1/2/3 = 250/250/750/750. |
| `DigitalLearningGmbH/MATH-lighteval` train | 1,500 | 30% | Levels 1--5 = 300 rows each. |
| `open-thoughts/OpenThoughts-114k` metadata | 1,000 | 20% | Biology/chemistry/physics/puzzle = 250 rows each. |

Student rollouts use a maximum total sequence length of 16,384 tokens.
Trajectories that reach the limit without a natural EOS are discarded.

## Training strategy

Training uses a serial student-rollout, teacher-scoring, and student-update
pipeline:

1. The Q3 student samples a completion with thinking enabled,
   `reasoning_effort=medium`, temperature 1.0, `top_p=0.95`, and `top_k=20`.
2. The BF16 teacher is loaded separately and teacher-forced on the complete
   student trajectory: the original prompt followed by the student's
   completion. Teacher hidden states are collected for the full sequence so
   each completion-token distribution is conditioned on the complete prompt
   and preceding response.
3. The student is loaded with DoRA rank 8, alpha 16, and merge-consistent STE.
   The forward path uses `Q(W + delta W)`, and the backward path applies the STE
   identity gradient.
4. The objective is full-vocabulary forward KL,
   `KL(p_teacher || p_student)`, at temperature 1.0. The vocabulary projection
   is evaluated in token-position chunks. The completion mask excludes prompt
   positions from the KL loss; only student completion tokens are supervised.
5. The loss is the global mean over valid completion tokens in each gradient
   accumulation group.
6. The trained adapter is merged into the base weights, followed by
   requantization with the original AWQ clipping ratios.

| Setting | Value |
|---|---:|
| Epochs | 1 |
| Adapter | DoRA rank 8, alpha 16 |
| Quantized training path | merge-consistent STE |
| Optimizer | fused AdamW |
| Peak learning rate | `1e-4` |
| Weight decay | `0` |
| Warmup | 3% of optimizer steps |
| Scheduler | cosine decay |
| Gradient clipping | `1.0` |
| Gradient accumulation | 8 |
| Checkpoint cadence | every one-third of total optimizer steps and at shard handoffs |
| Checkpoint retention | 1 |

Rollouts are organized into 480-sample shards plus a final remainder. Each
shard is teacher-scored and consumed before moving to the next shard, while the
optimizer, scheduler, RNG state, telemetry, and global counters continue across
shard boundaries.

## Evaluation strategy

The Q3, OPD, and BF16 variants use the same benchmark inputs, prompt templates,
and generation settings:

- thinking enabled with `reasoning_effort=medium`;
- greedy decoding with temperature `0`, `top_p=1`, and seed 42;
- request concurrency 10;
- MATH-500 and HumanEval+ use a 16,384-token completion limit;
- GPQA Diamond uses a 32,768-token completion limit;
- truncated responses, empty final answers, exhausted retries, and sandbox
  failures receive a score of zero.

MATH-500 uses all 500 test examples and exact-answer pass@1 scoring. GPQA uses
all 198 examples from the Diamond split with instruction-format multiple-choice
pass@1 scoring. HumanEval and HumanEval+ use all 164 problems; generated code is
executed in the EvalPlus Docker sandbox. A HumanEval+ pass requires the
candidate to pass both the original HumanEval tests and the additional
HumanEval+ tests.
