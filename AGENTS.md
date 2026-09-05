# LatticeRun repository guidance

This file governs the entire repository. Keep it focused on durable project
rules, architectural boundaries, and verification requirements. Do not use it
as an experiment diary, status log, command transcript, or server runbook.

## Project scope

LatticeRun is a reusable framework for:

1. architecture-aware low-bit quantization;
2. capability recovery through on-policy distillation (OPD) with LoRA or DoRA;
3. merged and requantized export; and
4. packed low-memory inference runtimes.

Qwen3.8-27B dense is the default reference model, not a reason to place
Qwen-specific assumptions in generic code. Additional dense models should be
integrated through the model adapter and registry. MoE remains a future
extension and must not be pre-implemented through speculative abstractions.

## Repository boundaries

- Keep Python sources directly under `src/`; do not add a physical
  `src/latticerun/` wrapper. Packaging maps `src` to the logical
  `latticerun.*` namespace.
- `model/` owns architecture recognition, loading, module classification,
  precision policy, and model-specific prompt or processor behavior.
- `quant/` owns generic quantization, clipping, calibration, sensitivity, and
  export primitives. It may consume model decisions but must not encode model
  names or module-name rules.
- `adapters/` owns LoRA, DoRA, STE behavior, merge semantics, and recovery
  parameter serialization.
- `opd/` owns rollouts, teacher artifacts, loss computation, token reduction,
  checkpointing, telemetry, and training orchestration.
- `runtime/` owns packed-weight execution, kernels, memory planning, and
  offload. Model-specific runtime policy must enter through the model adapter.
- Keep the CLI thin: parse configuration, resolve the model adapter, and call
  package APIs.
- Generic modules may depend on shared model contracts, but must not import a
  concrete adapter such as the Qwen adapter.

## Quantization and recovery invariants

- The default reference checkpoint is the original BF16 model. Do not quantize
  an already quantized checkpoint unless an experiment explicitly studies
  double quantization.
- The baseline weight format is symmetric INT3, group size 128, zero point 0,
  with signed fake-quantization codes in `[-3, 3]` and a zero-scale guard.
- The initial Qwen policy protects numerically sensitive control paths and uses
  INT4 for `lm_head`; this policy belongs in the Qwen model adapter.
- AWQ-style output-MSE clipping and weight-MSE clipping share the locked ratio
  grid `(1.00, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55)` unless a
  separately reported ablation changes it.
- Preserve block-output sensitivity measurement and the auditable codes/scales
  export path when refactoring.
- Student rollouts define the OPD trajectories. Teacher forcing evaluates the
  complete prompt-plus-student-completion sequence; the KL mask applies only to
  valid completion tokens.
- The reference objective is exact full-vocabulary forward KL at temperature
  1.0, reduced as a global mean over valid completion tokens. Token chunking may
  reduce memory but must remain mathematically equivalent to the unchunked
  objective.
- The production recovery path is merge-consistent DoRA plus STE; retain LoRA
  as a parallel supported adapter. Recovery dropout is not part of the current
  API.
- Export must materialize the effective recovered weight, merge the adapter,
  requantize with the selected policy, and preserve the metadata required by a
  real packed low-bit runtime.

## Multimodal policy

- PTQ and OPD remain text-only by default so the vision tower does not consume
  training memory or enter text-only sensitivity and recovery decisions.
- Deployment may optionally attach the model's original BF16 vision stack as a
  sidecar. Preserve the complete visual path, including patch embedding,
  vision blocks, merger/projector, processor configuration, and special-token
  behavior; an encoder without its bridge to the language model is not a valid
  multimodal implementation.
- Vision loading must be explicit and lazy. Text-only deployment must not pay
  its VRAM or initialization cost.
- Treat image understanding and computer-use control as separate layers.
  Computer use also requires screenshot processing, coordinate/action
  conventions, tool execution, and an agent loop.

## Public and local content

- Public `main` contains reusable framework code, generic tests, the license,
  and concise public documentation. Do not add provider-specific SSH helpers,
  one-off dataset downloaders, benchmark queues, local paths, or experiment
  artifacts.
- `feat/runtime` is the public runtime development line. Preserve its packed
  checkpoint foundation when integrating runtime work.
- `.agent/` is ignored local state. Its `Agent.md` may retain the historical
  experiment setup, infrastructure notes, and handoff record, but none of that
  content is normative for public framework behavior.
- Keep model weights, datasets, checkpoints, runs, results, caches, secrets,
  and machine-specific files outside Git.
- Experimental reference documentation may be developed on a dedicated docs
  branch and must be reviewed before it is committed or merged.

## Engineering and verification

- Preserve validated numerical behavior during refactors; improve ownership
  and interfaces without silently simplifying the algorithms.
- Add or update tests in the matching `tests/model`, `tests/quant`,
  `tests/adapter`, or `tests/opd` directory.
- Test reference equivalence for quantization, DoRA/LoRA merge, STE gradients,
  KL chunking and masking, checkpoint resume, device placement, and final
  requantized export whenever those paths change.
- Run the focused tests first, then the full suite and lint/format checks before
  committing a coherent change.
- Do not commit or push unrelated user changes. Do not rewrite public history
  unless the user explicitly authorizes it.
