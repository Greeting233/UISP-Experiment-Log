"""
src/random_multiseed.py
================================================================================
Random baseline multi-seed validation.

目的:
  量化 Random gating 的 seed 方差, 校准 Magnitude/EKF 相对 Random 领先幅度的
  置信度. 回答两个问题:
    1) "uncertainty-driven quantile + soft gate" 是否稳定优于 Random? (核心)
    2) v4 报告的 Δ=+3.34% 是真信号还是 Random 单次 seed 运气?

依赖:
  从同目录 inference_ablation.py import 所有 gate/wrapper/loader, 
  保证配置与消融实验严格一致.

配置 (与 v4 / ablation_v1 完全一致):
  q_min=0.1, q_max=0.3, alpha=3.0, warmup=5, batch=128
  EKF: Q=1e-3, R=1e-2
  Random seeds: 0, 1, ..., N-1 (默认 N=10)

Usage:
    %cd /content/drive/MyDrive/ekf_adaptive_pruning
    !python src/random_multiseed.py
    !python src/random_multiseed.py --n_seeds 20     # 想要更稳可以加

Output:
    results/random_multiseed.json   (每个 seed 原始数据 + 统计量)
    results/random_multiseed.png    (可视化: Random 分布 vs Mag/EKF 位置)
    stdout                          (summary + z-score + 判定)
================================================================================
"""

import os
import sys
import json
import copy
import argparse

import numpy as np
import torch

# 自己所在目录 (src/) 加入 sys.path, 以便 import sibling module
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
# 项目根 (上一级) 加入 sys.path, 以便 inference_ablation 能 import models/
_PROJ_ROOT = os.path.dirname(_THIS_DIR)
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

# 复用消融脚本里的所有组件 (保证配置一致)
from inference_ablation import (
    EKFGate, MagnitudeGate, RandomGate, GatedResNet,
    load_base_model, get_test_loader, evaluate,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default='./checkpoints/resnet18_cifar_base.pth')
    parser.add_argument('--data', default='./data')
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--q_min', type=float, default=0.1)
    parser.add_argument('--q_max', type=float, default=0.3)
    parser.add_argument('--alpha', type=float, default=3.0)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--Q', type=float, default=1e-3)
    parser.add_argument('--R', type=float, default=1e-2)
    parser.add_argument('--n_seeds', type=int, default=10,
                        help='Random seeds to run (seeds = 0, 1, ..., N-1)')
    parser.add_argument('--out_dir', default='./results')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[env] device={device}  torch={torch.__version__}")
    os.makedirs(args.out_dir, exist_ok=True)

    loader = get_test_loader(args.data, args.batch)
    base = load_base_model(device, args.ckpt)

    common_kwargs = dict(
        q_min=args.q_min, q_max=args.q_max,
        alpha=args.alpha, warmup_steps=args.warmup
    )

    results = {'config': vars(args)}

    # ================================================================
    # Sanity check: baseline acc 必须 ≈ 94.97% (确认 ckpt 加载正确)
    # ================================================================
    print("\n=== Sanity check: No pruning ===")
    acc, _, _ = evaluate(base, loader, device, track_keep=False)
    print(f"  acc = {acc*100:.2f}%")
    if acc < 0.90:
        print("  [FATAL] baseline 加载仍失败, 检查 checkpoint 结构")
        return
    results['no_prune_acc'] = acc

    # ================================================================
    # EKF reference (single run, seed-independent 验证)
    # ================================================================
    print("\n=== Reference: EKF gating (single run) ===")
    m = GatedResNet(
        copy.deepcopy(base), EKFGate,
        Q=args.Q, R=args.R, **common_kwargs
    ).to(device)
    acc, keep, per = evaluate(m, loader, device)
    results['ekf'] = {'acc': acc, 'keep': keep, 'per_gate_keep': per.tolist()}
    print(f"  acc = {acc*100:.2f}%,  avg keep = {keep:.4f}")

    # ================================================================
    # Magnitude reference (single run, seed-independent)
    # ================================================================
    print("\n=== Reference: Magnitude |γ| gating (single run) ===")
    m = GatedResNet(
        copy.deepcopy(base), MagnitudeGate, **common_kwargs
    ).to(device)
    acc, keep, per = evaluate(m, loader, device)
    results['magnitude'] = {'acc': acc, 'keep': keep, 'per_gate_keep': per.tolist()}
    print(f"  acc = {acc*100:.2f}%,  avg keep = {keep:.4f}")

    # ================================================================
    # Random multi-seed
    # ================================================================
    print(f"\n=== Random multi-seed (N = {args.n_seeds} seeds: 0..{args.n_seeds-1}) ===")
    random_runs = []
    for s in range(args.n_seeds):
        m = GatedResNet(
            copy.deepcopy(base), RandomGate,
            seed=s, **common_kwargs
        ).to(device)
        acc, keep, per = evaluate(m, loader, device)
        random_runs.append({
            'seed': s,
            'acc': acc,
            'keep': keep,
            'per_gate_keep': per.tolist(),
        })
        print(f"  seed={s:2d}:  acc={acc*100:.2f}%,  keep={keep:.4f}")

    results['random_runs'] = random_runs

    # ================================================================
    # Statistics
    # ================================================================
    accs = np.array([r['acc'] for r in random_runs])
    keeps = np.array([r['keep'] for r in random_runs])

    # ddof=1 是样本标准差 (unbiased)
    rand_stats = {
        'n':      int(len(accs)),
        'mean':   float(accs.mean()),
        'std':    float(accs.std(ddof=1)),
        'min':    float(accs.min()),
        'max':    float(accs.max()),
        'median': float(np.median(accs)),
        'keep_mean': float(keeps.mean()),
        'worst_seed': int(accs.argmin()),
        'best_seed':  int(accs.argmax()),
    }
    results['random_stats'] = rand_stats

    # ================================================================
    # Summary + z-score 判定
    # ================================================================
    print("\n" + "=" * 72)
    print("                    RANDOM MULTI-SEED SUMMARY")
    print("=" * 72)
    print(f"  No pruning      : {results['no_prune_acc']*100:.2f}%")
    print(f"  EKF       (1×)  : {results['ekf']['acc']*100:.2f}%   "
          f"keep = {results['ekf']['keep']:.4f}")
    print(f"  Magnitude (1×)  : {results['magnitude']['acc']*100:.2f}%   "
          f"keep = {results['magnitude']['keep']:.4f}")
    print(f"  Random   (N={rand_stats['n']:2d}) : "
          f"{rand_stats['mean']*100:.2f}% ± {rand_stats['std']*100:.2f}%   "
          f"keep = {rand_stats['keep_mean']:.4f}")
    print(f"     distribution : "
          f"min={rand_stats['min']*100:.2f}%  "
          f"median={rand_stats['median']*100:.2f}%  "
          f"max={rand_stats['max']*100:.2f}%")
    print(f"     worst seed   : {rand_stats['worst_seed']} "
          f"({rand_stats['min']*100:.2f}%)")
    print(f"     best seed    : {rand_stats['best_seed']} "
          f"({rand_stats['max']*100:.2f}%)")
    print("-" * 72)

    mag_acc = results['magnitude']['acc']
    rand_mean = rand_stats['mean']
    rand_std = rand_stats['std']
    delta_mean = (mag_acc - rand_mean) * 100
    z = (mag_acc - rand_mean) / max(rand_std, 1e-8)
    # 如果 Mag 在 Random 所有样本之上, 报告 gap_to_max
    gap_to_max = (mag_acc - rand_stats['max']) * 100

    print(f"  Δ Mag − Rand.mean = {delta_mean:+.2f}%")
    print(f"  Δ Mag − Rand.max  = {gap_to_max:+.2f}%   "
          f"(是否高于所有 Random seed?)")
    print(f"  z-score           = {z:+.2f}   "
          f"(Mag 相对 Random 分布偏离 σ 的倍数)")

    results['assessment'] = {
        'delta_mag_rand_mean': delta_mean,
        'delta_mag_rand_max':  gap_to_max,
        'z_score': float(z),
    }

    print("\n=== Assessment ===")
    if gap_to_max > 0 and z > 3:
        verdict = "STRONG"
        print(f"  [STRONG] Mag > Rand.max 且 z={z:.2f} > 3")
        print(f"    => 'quantile + soft gate + uncertainty' 机制显著有效")
        print(f"    => v4 pipeline 底座成立, 可放心进入 v5")
    elif z > 2:
        verdict = "MODERATE"
        print(f"  [MODERATE] z={z:.2f} ∈ (2, 3]")
        print(f"    => 机制有效但 margin 不算宽, v5 对比建议带 error bar")
    elif z > 1:
        verdict = "WEAK"
        print(f"  [WEAK] z={z:.2f} ∈ (1, 2]")
        print(f"    => 机制的独立价值不够硬, v5 必须切观测源才能讲故事")
    else:
        verdict = "NULL"
        print(f"  [NULL] z={z:.2f} ≤ 1")
        print(f"    => v4 的 +3.34% 可能主要来自 Random 的单次运气")
        print(f"    => side project 当前状态仅是原型, 不是可 defend 的结论")
    results['assessment']['verdict'] = verdict
    print("=" * 72)

    # ================================================================
    # Plot: Random distribution vs Mag/EKF 位置
    # ================================================================
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 5))

        # Random: 每个 seed 画一个点
        xs = np.arange(rand_stats['n'])
        ax.scatter(xs, accs * 100, s=60, color='#555', alpha=0.75,
                   zorder=3, label='Random (per seed)')

        # Random mean + ±1σ 带
        ax.axhline(rand_mean * 100, color='#555', linestyle='--',
                   linewidth=1.2, alpha=0.7,
                   label=f"Random mean = {rand_mean*100:.2f}%")
        ax.fill_between(
            [-0.5, rand_stats['n'] - 0.5],
            (rand_mean - rand_std) * 100, (rand_mean + rand_std) * 100,
            color='#888', alpha=0.15,
            label=f"Random ±1σ ({rand_std*100:.2f}%)"
        )

        # Mag / EKF 水平线
        ax.axhline(mag_acc * 100, color='#2ca02c', linewidth=2.5,
                   label=f"Magnitude = {mag_acc*100:.2f}%")
        ax.axhline(results['ekf']['acc'] * 100, color='#1f77b4',
                   linewidth=1.5, linestyle=':',
                   label=f"EKF = {results['ekf']['acc']*100:.2f}%")
        # No prune 作为 ceiling
        ax.axhline(results['no_prune_acc'] * 100, color='#d62728',
                   linewidth=1.2, linestyle='-', alpha=0.5,
                   label=f"No prune = {results['no_prune_acc']*100:.2f}%")

        ax.set_xlabel('Random seed index')
        ax.set_ylabel('Test accuracy (%)')
        ax.set_title(
            f"Random multi-seed validation  (N={rand_stats['n']})\n"
            f"Δ Mag − Rand.mean = {delta_mean:+.2f}%    "
            f"z = {z:.2f}    "
            f"verdict: {verdict}"
        )
        ax.legend(loc='lower left', fontsize=9, framealpha=0.92)
        ax.grid(alpha=0.3, zorder=1)
        ax.set_xticks(xs)
        ax.set_xlim(-0.5, rand_stats['n'] - 0.5)

        plt.tight_layout()
        png_path = os.path.join(args.out_dir, 'random_multiseed.png')
        plt.savefig(png_path, dpi=140)
        plt.close()
        print(f"\n[plot] saved to {png_path}")
    except Exception as e:
        print(f"\n[plot] skipped: {type(e).__name__}: {e}")

    # ================================================================
    # Save JSON
    # ================================================================
    out_path = os.path.join(args.out_dir, 'random_multiseed.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"[save] {out_path}")


if __name__ == '__main__':
    main()
