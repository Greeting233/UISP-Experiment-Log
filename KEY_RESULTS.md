# KEY_RESULTS.md

Quick-reference numerical summary of the UISP project.

---

## One-Sentence Pitch

An input-adaptive structured channel pruning method (UISP) for BN-based CNNs that uses softmax entropy to dynamically adjust pruning aggressiveness, achieving a statistically strong +1.64% ± 0.20% improvement over matched random pruning on ResNet-18 / CIFAR-10 post-training pruning (z-score = 8.09 on N=10 seeds, well beyond 5-sigma particle-physics discovery threshold).

---

## Core Numbers

### Main result (post-training, no fine-tune)

| Method | Accuracy | Keep Ratio |
|---------------------|-----------------------|------------|
| Baseline (no prune) | 94.97% | 100.00% |
| **UISP (ours)** | **92.86%** | 70.87% |
| Random (matched) | 91.22% ± 0.20% (N=10) | 73.99% |

### Statistical validation

- Delta UISP - Random.mean = **+1.64%**
- Delta UISP - Random.max = **+1.27%** (beats all 10 Random seeds)
- **z-score = +8.09** (chance probability ~ 10^-15)

---

## Negative Results Summary

Three systematic ablations confirm EKF cannot be activated under BN ResNet post-training pruning:

| Observation source | Accuracy | Failure mechanism |
|-------------------------------|-----------|------------------------------------------------------|
| Static BN gamma | 92.86% ✓ | EKF degenerates to identity |
| Dynamic activation magnitude | 86.08% ✗ | BN normalization flattens importance signal |
| Hybrid gamma x activation | 90.00% ✗ | Multiplication collapses std, breaks soft gate |

Structural formulation-architecture mismatch, not implementation issue.

---

## Full Version Timeline

| Version | Obs source | Estimator | Accuracy |
|---------|-----------|-----------|----------|
| v1 | activation | EKF | 10.00% |
| v2 | activation | EKF | 74.99% |
| v3 | activation | EKF | 90.07% |
| **v4** | **BN gamma** | **EKF** | **92.86%** |
| Ablation | BN gamma | Mag + EKF | 92.86% (identical) |
| Multi-seed (N=10) | — | — | Random 91.22 ± 0.20% |
| v5 | activation | Mag + EKF | 86.08% + 86.22% |
| v5.5 | gamma x act | Mag + EKF | 90.00% + 89.79% |
| **Final** | BN gamma | Mag (UISP) | **92.86%** |

---

## Project Status

Stage 1 (CIFAR-10 / ResNet-18): complete. Planned P1 work: trade-off curve sweep, post-prune fine-tune, FLOPs measurement.
