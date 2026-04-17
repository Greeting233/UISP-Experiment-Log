"""
src/inference_v5.py
================================================================================
v5: 2×2 factor design × Random multi-seed baseline.

观测源 (obs_source):
  - bn_gamma   : z = |γ_i|  (learned parameter, inference 恒定)
  - activation : z = F.relu(feat).abs().mean(dim=(0,2,3))  (batch-to-batch 动态)

估计方式 (estimator):
  - magnitude  : θ̂ := z  (直接)
  - ekf        : θ̂ = EKF 对角递推(z)

2×2 cells:
  - Mag-γ      (v4 Magnitude 复现)
  - EKF-γ      (v4 EKF 复现, 应 = Mag-γ 退化验证)
  - Mag-act    ← 新: 观测源切换的纯效应
  - EKF-act    ← ★ 新: EKF 在动态观测下的真正主角

+ Random × N seeds  (fair baseline)

KEY DELTAS:
  Δ Mag-act − Mag-γ       : 切换观测源本身的增益
  Δ EKF-γ   − Mag-γ       : γ 下 EKF 贡献 (应 ≈ 0, 退化验证)
  Δ EKF-act − Mag-act     : ★ EKF 在动态观测下的增量 (CORE)
  Δ EKF-act − Rand.mean   : end-to-end 领先度

Usage:
    %cd /content/drive/MyDrive/ekf_adaptive_pruning
    !python src/inference_v5.py                      # 默认 10 seeds
    !python src/inference_v5.py --n_seeds 20         # 更稳

Output:
    results/v5_factor_design.png
    results/v5_results.json
================================================================================
"""

import os
import sys
import math
import json
import copy
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# sibling + project root 加入 sys.path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
_PROJ_ROOT = os.path.dirname(_THIS_DIR)
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

# 从 ablation 脚本复用 data loader + model loader + eval (保证 pipeline 一致)
from inference_ablation import (
    load_base_model,
    get_test_loader,
    evaluate,
)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# ================================================================
# Gate 实现 (v5 统一版, 通过 obs_source 参数切换观测源)
# ================================================================

class _BaseGate(nn.Module):
    """
    公共基类. 
    - obs_source 决定 _read_observation(x) 行为
    - 子类实现 _compute_theta(z)
    """
    VALID_OBS = ('bn_gamma', 'activation')

    def __init__(self, bn_layer, num_channels,
                 obs_source='bn_gamma',
                 q_min=0.1, q_max=0.3, alpha=3.0,
                 warmup_steps=5, entropy_norm=None):
        super().__init__()
        if obs_source not in self.VALID_OBS:
            raise ValueError(f"obs_source must be one of {self.VALID_OBS}, got {obs_source}")
        self.bn_layer = bn_layer
        self.num_channels = num_channels
        self.obs_source = obs_source
        self.q_min = q_min
        self.q_max = q_max
        self.alpha = alpha
        self.warmup_steps = warmup_steps
        self.entropy_norm = entropy_norm if entropy_norm is not None else math.log(10.0)

        self.register_buffer('step_count', torch.zeros(1, dtype=torch.long))
        self._logits_prev = None
        self.last_keep_ratio = 1.0
        # 诊断: 记录最后一次 observation 的 channel-wise std
        # (不能直接告诉我们 batch-to-batch 是否变, 但能看出 channel 间异质性)
        self.last_obs_std = None

    def set_context(self, logits_prev):
        self._logits_prev = logits_prev

    def _uncertainty(self):
        if self._logits_prev is None:
            return 0.5
        p = F.softmax(self._logits_prev, dim=-1)
        ent = -(p * torch.log(p + 1e-8)).sum(-1).mean()
        return float(torch.clamp(ent / self.entropy_norm, 0.0, 1.0).item())

    def _read_observation(self, x):
        """
        x: [B, C, H, W], BasicBlock 的 post-ReLU 输出.
        根据 obs_source 返回 [C] 的观测量 z.
        """
        if self.obs_source == 'bn_gamma':
            # 静态: BN scale parameter
            return self.bn_layer.weight.detach().abs()
        elif self.obs_source == 'activation':
            # 动态: 用户选定方案
            # F.relu(feat).abs().mean((0,2,3))
            # 注意: 因为 x 已经是 post-ReLU, 这里 F.relu 是冗余的 no-op,
            #       但保留以匹配接口规范(未来若换挂载点仍稳健).
            return F.relu(x).abs().mean(dim=(0, 2, 3)).detach()
        else:
            raise ValueError(self.obs_source)

    def _compute_theta(self, z):
        raise NotImplementedError

    def forward(self, x):
        self.step_count += 1

        z = self._read_observation(x)
        self.last_obs_std = float(z.std().item())

        theta = self._compute_theta(z)

        if self.step_count.item() <= self.warmup_steps:
            self.last_keep_ratio = 1.0
            return x

        u = self._uncertainty()
        q = self.q_min + (self.q_max - self.q_min) * (1.0 - u)
        tau = torch.quantile(theta, q)
        std = theta.std() + 1e-8
        gate = torch.sigmoid(self.alpha * (theta - tau) / std)

        self.last_keep_ratio = float(gate.mean().item())
        return x * gate.view(1, -1, 1, 1)


class MagnitudeGate(_BaseGate):
    """θ̂ := z 直接"""
    def _compute_theta(self, z):
        return z


class EKFGate(_BaseGate):
    """θ̂ = EKF(z) 对角递推"""
    def __init__(self, bn_layer, num_channels, Q=1e-3, R=1e-2, **kwargs):
        super().__init__(bn_layer, num_channels, **kwargs)
        self.Q = Q
        self.R = R
        self.register_buffer('theta_hat', torch.zeros(num_channels))
        self.register_buffer('P', torch.ones(num_channels))
        self.register_buffer('initialized', torch.zeros(1, dtype=torch.bool))

    def _compute_theta(self, z):
        if not self.initialized.item():
            self.theta_hat.copy_(z)
            self.initialized.fill_(True)
        else:
            P_pred = self.P + self.Q
            K = P_pred / (P_pred + self.R)
            self.theta_hat.add_(K * (z - self.theta_hat))
            self.P.copy_((1.0 - K) * P_pred)
        return self.theta_hat


class RandomGate(nn.Module):
    """Matched-keep-ratio random gating. 不使用 obs_source."""
    def __init__(self, bn_layer, num_channels,
                 q_min=0.1, q_max=0.3, alpha=3.0,
                 warmup_steps=5, entropy_norm=None, seed=42,
                 obs_source=None):  # 接收但忽略 (接口统一)
        super().__init__()
        self.bn_layer = bn_layer
        self.num_channels = num_channels
        self.q_min = q_min
        self.q_max = q_max
        self.warmup_steps = warmup_steps
        self.entropy_norm = entropy_norm if entropy_norm is not None else math.log(10.0)

        self.register_buffer('step_count', torch.zeros(1, dtype=torch.long))
        self._logits_prev = None
        self.last_keep_ratio = 1.0
        self.last_obs_std = None
        self.gen = torch.Generator()
        self.gen.manual_seed(seed)

    def set_context(self, logits_prev):
        self._logits_prev = logits_prev

    def _uncertainty(self):
        if self._logits_prev is None:
            return 0.5
        p = F.softmax(self._logits_prev, dim=-1)
        ent = -(p * torch.log(p + 1e-8)).sum(-1).mean()
        return float(torch.clamp(ent / self.entropy_norm, 0.0, 1.0).item())

    def forward(self, x):
        self.step_count += 1
        if self.step_count.item() <= self.warmup_steps:
            self.last_keep_ratio = 1.0
            return x

        u = self._uncertainty()
        q = self.q_min + (self.q_max - self.q_min) * (1.0 - u)
        num_prune = int(round(self.num_channels * q))

        perm = torch.randperm(self.num_channels, generator=self.gen)
        prune_idx = perm[:num_prune].to(x.device)
        mask = torch.ones(self.num_channels, device=x.device)
        mask[prune_idx] = 0.0

        self.last_keep_ratio = float(mask.mean().item())
        return x * mask.view(1, -1, 1, 1)


# ================================================================
# GatedResNet wrapper
# ================================================================

class GatedResNet(nn.Module):
    def __init__(self, base_model, gate_class, **gate_kwargs):
        super().__init__()
        self.base = base_model
        self.gates = nn.ModuleList()
        self._logits_prev = None

        for layer_name in ['layer3', 'layer4']:
            for block in getattr(self.base, layer_name):
                bn = block.bn2
                C = bn.num_features
                gate = gate_class(bn_layer=bn, num_channels=C, **gate_kwargs)
                self.gates.append(gate)
                self._attach_hook(block, gate)

    @staticmethod
    def _attach_hook(block, gate):
        def hook(module, inp, out):
            return gate(out)
        block.register_forward_hook(hook)

    def forward(self, x):
        for g in self.gates:
            g.set_context(self._logits_prev)
        logits = self.base(x)
        self._logits_prev = logits.detach()
        return logits

    def get_keep_ratios(self):
        return [g.last_keep_ratio for g in self.gates]

    def get_obs_stds(self):
        return [g.last_obs_std for g in self.gates]


# ================================================================
# Run helper
# ================================================================

def run_one(name, base, loader, device, gate_class, common_kwargs, **extra):
    m = GatedResNet(
        copy.deepcopy(base), gate_class,
        **common_kwargs, **extra
    ).to(device)
    acc, keep, per = evaluate(m, loader, device)
    obs_stds = m.get_obs_stds()
    return {
        'name': name,
        'acc': acc,
        'keep': keep,
        'per_gate_keep': per.tolist(),
        'per_gate_obs_std_final': [float(s) if s is not None else None for s in obs_stds],
    }


# ================================================================
# Main
# ================================================================

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
    parser.add_argument('--n_seeds', type=int, default=10)
    parser.add_argument('--out_dir', default='./results')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[env] device={device}  torch={torch.__version__}")
    os.makedirs(args.out_dir, exist_ok=True)

    loader = get_test_loader(args.data, args.batch)
    base = load_base_model(device, args.ckpt)

    common_kwargs = dict(
        q_min=args.q_min, q_max=args.q_max,
        alpha=args.alpha, warmup_steps=args.warmup,
    )

    results = {'config': vars(args)}

    # === [0] Sanity ===
    print("\n=== [0] Sanity: No pruning ===")
    acc, _, _ = evaluate(base, loader, device, track_keep=False)
    results['no_prune'] = {'acc': acc}
    print(f"  acc = {acc*100:.2f}%")
    if acc < 0.90:
        print("  [FATAL] baseline 加载失败"); return

    # === [2x2 factor] ===
    cells = []

    print("\n=== [2x2] Mag-γ  (obs=bn_gamma, est=magnitude)  [v4 reference] ===")
    r = run_one('mag_gamma', base, loader, device, MagnitudeGate, common_kwargs,
                obs_source='bn_gamma')
    print(f"  acc = {r['acc']*100:.2f}%, keep = {r['keep']:.4f}")
    print(f"  per-gate obs std (channel-wise) = "
          f"{[round(s, 5) for s in r['per_gate_obs_std_final']]}")
    cells.append(r)

    print("\n=== [2x2] EKF-γ  (obs=bn_gamma, est=ekf)  [退化验证, 应 = Mag-γ] ===")
    r = run_one('ekf_gamma', base, loader, device, EKFGate, common_kwargs,
                obs_source='bn_gamma', Q=args.Q, R=args.R)
    print(f"  acc = {r['acc']*100:.2f}%, keep = {r['keep']:.4f}")
    cells.append(r)

    print("\n=== [2x2] Mag-act  (obs=activation, est=magnitude)  [NEW] ===")
    r = run_one('mag_act', base, loader, device, MagnitudeGate, common_kwargs,
                obs_source='activation')
    print(f"  acc = {r['acc']*100:.2f}%, keep = {r['keep']:.4f}")
    print(f"  per-gate obs std (channel-wise) = "
          f"{[round(s, 5) for s in r['per_gate_obs_std_final']]}")
    cells.append(r)

    print("\n=== [2x2] EKF-act  (obs=activation, est=ekf)  [★ v5 MAIN] ===")
    r = run_one('ekf_act', base, loader, device, EKFGate, common_kwargs,
                obs_source='activation', Q=args.Q, R=args.R)
    print(f"  acc = {r['acc']*100:.2f}%, keep = {r['keep']:.4f}")
    cells.append(r)

    results['cells'] = cells
    d = {c['name']: c for c in cells}

    # === Random multi-seed ===
    print(f"\n=== Random multi-seed (N={args.n_seeds}) ===")
    rand_runs = []
    for s in range(args.n_seeds):
        r = run_one(f'random_s{s}', base, loader, device, RandomGate,
                    common_kwargs, seed=s)
        rand_runs.append(r)
        print(f"  seed={s:2d}:  acc={r['acc']*100:.2f}%, keep={r['keep']:.4f}")

    results['random_runs'] = rand_runs
    rand_accs = np.array([r['acc'] for r in rand_runs])
    rand_stats = {
        'n':      int(len(rand_accs)),
        'mean':   float(rand_accs.mean()),
        'std':    float(rand_accs.std(ddof=1)),
        'min':    float(rand_accs.min()),
        'max':    float(rand_accs.max()),
        'median': float(np.median(rand_accs)),
    }
    results['random_stats'] = rand_stats

    # === Summary ===
    print("\n" + "=" * 78)
    print("                    v5: 2×2 FACTOR + Random N  SUMMARY")
    print("=" * 78)
    print(f"{'Method':<14}{'obs':<14}{'estimator':<14}{'Acc':>9}{'Keep':>9}")
    print("-" * 78)
    print(f"{'No pruning':<14}{'--':<14}{'--':<14}"
          f"{results['no_prune']['acc']*100:>8.2f}%{'100.00%':>9}")
    cell_labels = [
        ('mag_gamma', 'bn_gamma',   'magnitude'),
        ('ekf_gamma', 'bn_gamma',   'ekf'),
        ('mag_act',   'activation', 'magnitude'),
        ('ekf_act',   'activation', 'ekf'),
    ]
    for name, obs, est in cell_labels:
        c = d[name]
        marker = '  ←★' if name == 'ekf_act' else ''
        print(f"{name:<14}{obs:<14}{est:<14}"
              f"{c['acc']*100:>8.2f}%{c['keep']*100:>8.2f}%{marker}")
    print(f"{'Random ('+str(rand_stats['n'])+' seeds)':<14}{'--':<14}{'--':<14}"
          f"{rand_stats['mean']*100:>6.2f}±{rand_stats['std']*100:.2f}%      --")
    print("-" * 78)

    # Deltas
    d_obsswitch    = (d['mag_act']['acc']  - d['mag_gamma']['acc']) * 100
    d_ekf_on_gamma = (d['ekf_gamma']['acc']- d['mag_gamma']['acc']) * 100
    d_ekf_on_act   = (d['ekf_act']['acc']  - d['mag_act']['acc'])   * 100
    d_main_rand    = (d['ekf_act']['acc']  - rand_stats['mean'])    * 100
    z_ekf_act = (d['ekf_act']['acc'] - rand_stats['mean']) / max(rand_stats['std'], 1e-8)
    z_mag_act = (d['mag_act']['acc'] - rand_stats['mean']) / max(rand_stats['std'], 1e-8)

    print("\n=== KEY DELTAS ===")
    print(f"  Δ Mag-act  − Mag-γ     = {d_obsswitch:>+6.2f}%     "
          f"(切换观测源本身的纯效应)")
    print(f"  Δ EKF-γ    − Mag-γ     = {d_ekf_on_gamma:>+6.2f}%     "
          f"(γ 观测下 EKF 贡献, 期望 ≈ 0)")
    print(f"  Δ EKF-act  − Mag-act   = {d_ekf_on_act:>+6.2f}%     "
          f"★ EKF 在动态观测下的增量 (CORE)")
    print(f"  Δ EKF-act  − Rand.mean = {d_main_rand:>+6.2f}%     "
          f"(z = {z_ekf_act:.2f})")
    print(f"  Δ Mag-act  − Rand.mean = "
          f"{(d['mag_act']['acc']-rand_stats['mean'])*100:>+6.2f}%     "
          f"(z = {z_mag_act:.2f})")

    results['key_deltas'] = {
        'obs_switch_mag_act_vs_mag_gamma': d_obsswitch,
        'ekf_on_gamma_degenerate_check':   d_ekf_on_gamma,
        'ekf_on_act_core_signal':          d_ekf_on_act,
        'ekf_act_vs_rand':                 d_main_rand,
        'z_ekf_act':                       float(z_ekf_act),
        'z_mag_act':                       float(z_mag_act),
    }

    # === Verdict on EKF ===
    print("\n=== Verdict on EKF (在动态观测下) ===")
    if d_ekf_on_act >= 0.5:
        verdict = 'EKF_ALIVE'
        print(f"  [EKF ALIVE]  Δ = {d_ekf_on_act:+.2f}% ≥ 0.5%")
        print(f"    => EKF 在 input-varying 观测下确实贡献时序滤波收益")
        print(f"    => 'EKF-guided Input-Adaptive' 叙事成立, 可保留项目命名")
        print(f"    => 下一步: Q/R grid 扫描找最优超参, 写 trade-off curve")
    elif abs(d_ekf_on_act) < 0.5:
        verdict = 'EKF_NEUTRAL'
        print(f"  [EKF NEUTRAL]  |Δ| = {abs(d_ekf_on_act):.2f}% < 0.5%")
        print(f"    => EKF 在当前 Q/R 默认值下仍无显著贡献")
        print(f"    => 下一步选项:")
        print(f"       (a) 扫 Q/R grid, 看是否存在 EKF 真正活跃的超参区间")
        print(f"       (b) 接受现状, side project 叙事转为")
        print(f"           'uncertainty-driven input-adaptive structured sparsification'")
        print(f"           (EKF 作为主 SCI 论文的专属机制, 不强行迁移)")
    else:
        verdict = 'EKF_HURT'
        print(f"  [EKF HURT]  Δ = {d_ekf_on_act:+.2f}% < -0.5%")
        print(f"    => EKF 过度平滑了动态信号, 反而损失时效性")
        print(f"    => 建议: 调小 R (观测更可信) 或调大 Q (状态变化更快)")
        print(f"    => 可以先试 --Q 1e-2 --R 1e-3 并重跑")
    results['verdict'] = verdict
    print("=" * 78)

    # === Plot ===
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10.5, 5.8))

        # Random dots + ±1σ band
        xs_r = np.arange(len(rand_accs))
        ax.scatter(xs_r, rand_accs * 100, s=55, color='#888', alpha=0.75,
                   zorder=3, label='Random (per seed)')
        ax.fill_between([-0.6, len(rand_accs)-0.4],
                        (rand_stats['mean']-rand_stats['std'])*100,
                        (rand_stats['mean']+rand_stats['std'])*100,
                        color='#888', alpha=0.14,
                        label=f"Random ±1σ ({rand_stats['std']*100:.2f}%)")
        ax.axhline(rand_stats['mean']*100, color='#888', linestyle='--',
                   linewidth=1, alpha=0.7,
                   label=f"Random mean = {rand_stats['mean']*100:.2f}%")

        # 2×2 cells as horizontal lines
        cell_style = {
            'mag_gamma': dict(color='#2ca02c', ls=':',  lw=1.3,
                              label=f"Mag-γ  = {d['mag_gamma']['acc']*100:.2f}%"),
            'ekf_gamma': dict(color='#1f77b4', ls=':',  lw=1.3,
                              label=f"EKF-γ  = {d['ekf_gamma']['acc']*100:.2f}%"),
            'mag_act':   dict(color='#ff7f0e', ls='-',  lw=1.9,
                              label=f"Mag-act = {d['mag_act']['acc']*100:.2f}%"),
            'ekf_act':   dict(color='#d62728', ls='-',  lw=2.6,
                              label=f"EKF-act = {d['ekf_act']['acc']*100:.2f}% ★"),
        }
        for name, style in cell_style.items():
            ax.axhline(d[name]['acc']*100, **style)

        # No-prune ceiling
        ax.axhline(results['no_prune']['acc']*100, color='k',
                   linewidth=0.8, linestyle='-', alpha=0.3,
                   label=f"No prune = {results['no_prune']['acc']*100:.2f}%")

        ax.set_xlabel('Random seed index')
        ax.set_ylabel('Test accuracy (%)')
        title = (
            f"v5: 2×2 factor design  (obs × estimator)  |  "
            f"Δ obs-switch = {d_obsswitch:+.2f}%  ·  "
            f"Δ EKF-on-act = {d_ekf_on_act:+.2f}%  ·  "
            f"verdict: {verdict}"
        )
        ax.set_title(title, fontsize=10)
        ax.legend(loc='lower right', fontsize=8, ncol=2, framealpha=0.92)
        ax.grid(alpha=0.3, zorder=1)
        ax.set_xticks(xs_r)
        ax.set_xlim(-0.6, len(rand_accs)-0.4)

        plt.tight_layout()
        png = os.path.join(args.out_dir, 'v5_factor_design.png')
        plt.savefig(png, dpi=140)
        plt.close()
        print(f"\n[plot] saved to {png}")
    except Exception as e:
        print(f"\n[plot] skipped: {type(e).__name__}: {e}")

    # === Save ===
    out = os.path.join(args.out_dir, 'v5_results.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"[save] {out}")


if __name__ == '__main__':
    main()
