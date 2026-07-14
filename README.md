# UISP: Uncertainty-Driven Input-Adaptive Structured Pruning

A boundary-diagnostic study of uncertainty-driven input-adaptive structured pruning for BN-based CNNs, on ResNet-18 / CIFAR-10 (post-training, no fine-tuning).

## What this is

The project tests whether a recursive, Kalman-style confidence proxy can drive input-adaptive channel pruning. The honest finding is a **negative / boundary result**: under static post-training BN weights, the confidence proxy degenerates to identity, and the method reduces to plain BN-γ magnitude gating. It does **not** claim a successful transfer of uncertainty-driven adaptation to this setting.

## Result (post-training, N=10 seeds)

| Method | Accuracy | Keep ratio |
|---|---|---|
| Baseline (no pruning) | 94.97% | 100% |
| Magnitude gating (UISP under static weights) | 92.86% | 70.9% |
| Matched random pruning | 91.22% ± 0.20% | 74.0% |

Magnitude gating beats matched random pruning by +1.64% (N=10); three ablations confirm the intended uncertainty mechanism cannot be activated under static BN weights. See `KEY_RESULTS.md` and `experiment_log.pdf` for the full numerical summary and ablations.

## Contents

- `KEY_RESULTS.md` — numerical summary and ablation table
- `experiment_log.pdf` — experiment log
- `train_base.py`, `inference_v5.py`, `inference_v5_5.py`, `inference_ablation.py`, `random_multiseed.py` — training / inference / ablation / matched-random-baseline scripts
- `notebooks/`, `figures/`

## License

MIT (see `LICENSE`).
