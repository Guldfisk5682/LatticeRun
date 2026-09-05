# LatticeRun

## About

LatticeRun is an open-source framework for low-bit quantization, capability
recovery, and efficient deployment of large language models. It separates
model-specific structure from reusable compression and distillation logic.

The workflow is:

```text
BF16 model
  -> calibration and low-bit quantization
  -> on-policy student rollouts
  -> BF16 teacher scoring on the same trajectories
  -> LoRA or DoRA capability recovery
  -> merge and requantization
  -> packed low-bit runtime
```

The framework provides symmetric groupwise quantization, AWQ-style
activation-aware clipping, weight-MSE clipping, block-output sensitivity,
merge-consistent STE recovery, exact full-vocabulary forward KL, global valid
completion-token reduction, resumable training, and auditable codes-and-scales
exports. Qwen3.8-27B dense is the default reference model.

## Getting Started

LatticeRun requires Python 3.11 or newer. Install the framework and training
dependencies in an isolated environment:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[train]'
```

Inspect the default model policy and run activation-aware quantization:

```bash
latticerun inspect --device cuda

latticerun quantize \
  --calibration-jsonl calibration.jsonl \
  --output artifacts/quantized
```

Capability recovery is an explicit serial pipeline so the student and BF16
teacher do not need to remain resident together:

```bash
latticerun opd rollout --prompts prompts.jsonl --output rollouts.jsonl
latticerun opd teacher --rollouts rollouts.jsonl --output teacher-artifacts
latticerun opd train \
  --teacher-artifacts teacher-artifacts \
  --clip-ratios artifacts/quantized/awq_clip_ratios.json \
  --output recovered-adapter
latticerun opd export \
  --adapter recovered-adapter \
  --clip-ratios artifacts/quantized/awq_clip_ratios.json \
  --output recovered-reference
```

Rollouts that reach the configured token ceiling without EOS are discarded.
The final export merges the recovery adapter, requantizes the effective
weights, and stores signed codes with FP32 scales for packed-runtime loading.

## Related Projects

- [vLLM](https://github.com/vllm-project/vllm) provides a high-throughput,
  memory-efficient serving engine and is an important reference for
  LatticeRun's runtime work.

## Citation

If LatticeRun is useful in your research, cite the repository and the exact
commit used for your experiments.
