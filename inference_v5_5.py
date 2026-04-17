"""
src/inference_v5_5.py
================================================================================
v5.5: Formulation salvage attempt with composite observation.

核心假设:
  纯 γ: 有序但静态 → EKF 退化
  纯 activation: 动态但无序 → pipeline 崩溃
  γ × activation: 结构先验 + 输入动态, 补两条路径各自的短板

新观测源:
    z_i = |γ_i| · mean_{b,h,w}|ReLU(y_i)|
    (gamma-weighted activation magnitude)

设计原则 (严格遵循用户指令):
  - 其余设置完全不变 (q_min/q_max, alpha, warmup, gate 形式, EKF Q/R)
  - 不扫 Q/R
  - 只看两个核心对比:
      (A) Mag-(γ×act) vs Mag-γ    : 组合观测避免了 activation-only 崩溃 ?
      (B) EKF-(γ×act) vs Mag-(γ×act) : EKF 终于获得非零增量 ?

判据 (formulation salvage 三档):
  情况 1: Mag-(γ×act) 明显差于 Mag-γ (≤ -2%) 或输给 Random
          → formulation salvage FAILED, 直接走 Z
  情况 2: Mag-(γ×act) 接近或超过 Mag-γ, 但 EKF ≈ Mag
          → dynamic obs 可以无害引入, 但 EKF 仍无用, 也建议 Z
  情况 3: Mag-(γ×act) 接近 Mag-γ, 且 EKF-(γ×act) − Mag-(γ×act) ≥ +0.8%
          → EKF 终于在有意义动态观测下活过来, 保留 EKF 叙事

Usage:
    %cd /content/drive/MyDrive/ekf_adaptive_pruning
    !python src/inference_v5_5.py

Output:
    results/v5_5_salvage.json
    results/v5_5_salvage.png
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

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
_PROJ_ROOT = os.path.dirname(_THIS_DIR)
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from inference_ablation import (
    load_base_model, get_test_loader, evaluate,
)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# ================================================================
# Gate: 在原 v5 _BaseGate 基础上新增 obs_source='gamma_times_activation'
# ================================================================

class _BaseGate(nn.Module):
    """与 v5 _BaseGate 一致, 但 _read_observation 新增组合模式."""
    VALID_OBS = ('bn_gamma', 'activation', 'gamma_times_activation')

    def __init__(self, bn_layer, num_channels,
                 obs_source='bn_gamma',
                 q_min=0.1, q_max=0.3, alpha=3.0,
                 warmup_steps=5, entropy_norm=None):
        super().__init__()
        if obs_source not in self.VALID_OBS:
            raise ValueError(f"obs_source must be in {self.VALID_OBS}")
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
        x: [B, C, H, W], BasicBlock 的 post-ReLU 输出

        新增 'gamma_times_activation':
            z_i = |γ_i| · mean_{b,h,w}|ReLU(y_i)|
        即 γ 提供结构先验排序, activation 提供输入动态调制.
        """
        if self.obs_source == 'bn_gamma':
            return self.bn_layer.weight.detach().abs()
        elif self.obs_source == 'activation':
            return F.relu(x).abs().mean(dim=(0, 2, 3)).detach()
        elif self.obs_source == 'gamma_times_activation':
            gamma_abs = self.bn_layer.weight.detach().abs()          # [C]
            act_mag = F.relu(x).abs().mean(dim=(0, 2, 3)).detach()   # [C]
            return gamma_abs * act_mag                                # [C]
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
    def _compute_theta(self, z):
        return z


class EKFGate(_BaseGate):
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
    def __init__(self, bn_layer, num_channels,
                 q_min=0.1, q_max=0.3, alpha=3.0,
                 warmup_steps=5, entropy_norm=None, seed=42,
                 obs_source=None):
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


def run_one(name, base, loader, device, gate_class, common_kwargs, **extra):
    m = GatedResNet(
        copy.deepcopy(base), gate_class,
        **common_kwargs, **extra
    ).to(device)
    acc, keep, per = evaluate(m, loader, device)
    return {
        'name': name,
        'acc': acc,
        'keep': keep,
        'per_gate_keep': per.tolist(),
        'per_gate_obs_std_final':
            [float(s) if s is not None else None for s in m.get_obs_stds()],
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

    # --- Sanity ---
    print("\n=== [0] Sanity: No pruning ===")
    acc, _, _ = evaluate(base, loader, device, track_keep=False)
    results['no_prune'] = {'acc': acc}
    print(f"  acc = {acc*100:.2f}%")
    if acc < 0.90:
        print("  [FATAL]"); return

    # --- Reference cells: Mag-γ (v4), Mag-act (v5) ---
    print("\n=== [ref-1] Mag-γ  (v4 reference) ===")
    r = run_one('mag_gamma', base, loader, device, MagnitudeGate, common_kwargs,
                obs_source='bn_gamma')
    print(f"  acc = {r['acc']*100:.2f}%, keep = {r['keep']:.4f}")
    print(f"  per-gate obs std = {[round(s,5) for s in r['per_gate_obs_std_final']]}")
    results['mag_gamma'] = r

    print("\n=== [ref-2] Mag-act  (v5 reference, 崩盘基准) ===")
    r = run_one('mag_act', base, loader, device, MagnitudeGate, common_kwargs,
                obs_source='activation')
    print(f"  acc = {r['acc']*100:.2f}%, keep = {r['keep']:.4f}")
    print(f"  per-gate obs std = {[round(s,5) for s in r['per_gate_obs_std_final']]}")
    results['mag_act'] = r

    # --- NEW cells: Mag-(γ×act), EKF-(γ×act) ---
    print("\n=== [★ v5.5] Mag-(γ×act)  (formulation salvage, 无 EKF) ===")
    r = run_one('mag_gamma_act', base, loader, device, MagnitudeGate, common_kwargs,
                obs_source='gamma_times_activation')
    print(f"  acc = {r['acc']*100:.2f}%, keep = {r['keep']:.4f}")
    print(f"  per-gate obs std = {[round(s,6) for s in r['per_gate_obs_std_final']]}")
    results['mag_gamma_act'] = r

    print("\n=== [★ v5.5] EKF-(γ×act)  (formulation salvage, EKF + 组合观测) ===")
    r = run_one('ekf_gamma_act', base, loader, device, EKFGate, common_kwargs,
                obs_source='gamma_times_activation', Q=args.Q, R=args.R)
    print(f"  acc = {r['acc']*100:.2f}%, keep = {r['keep']:.4f}")
    results['ekf_gamma_act'] = r

    # --- Random multi-seed (N=10) ---
    print(f"\n=== Random multi-seed (N={args.n_seeds}) ===")
    rand_runs = []
    for s in range(args.n_seeds):
        r = run_one(f'random_s{s}', base, loader, device, RandomGate,
                    common_kwargs, seed=s)
        rand_runs.append(r)
        print(f"  seed={s:2d}: acc={r['acc']*100:.2f}%, keep={r['keep']:.4f}")
    results['random_runs'] = rand_runs
    rand_accs = np.array([r['acc'] for r in rand_runs])
    rand_stats = {
        'mean': float(rand_accs.mean()),
        'std':  float(rand_accs.std(ddof=1)),
        'min':  float(rand_accs.min()),
        'max':  float(rand_accs.max()),
        'n':    int(len(rand_accs)),
    }
    results['random_stats'] = rand_stats

    # --- Summary ---
    print("\n" + "=" * 78)
    print("         v5.5  FORMULATION SALVAGE ATTEMPT  SUMMARY")
    print("=" * 78)
    print(f"{'Method':<22}{'Obs':<24}{'Est':<10}{'Acc':>9}{'Keep':>9}")
    print("-" * 78)
    print(f"{'No pruning':<22}{'--':<24}{'--':<10}"
          f"{results['no_prune']['acc']*100:>8.2f}%{'100.00%':>9}")
    rows = [
        ('mag_gamma',     'bn_gamma',               'direct'),
        ('mag_act',       'activation',             'direct'),
        ('mag_gamma_act', 'γ × activation (NEW)',   'direct'),
        ('ekf_gamma_act', 'γ × activation (NEW)',   'ekf'),
    ]
    for key, obs, est in rows:
        c = results[key]
        marker = '  ←★' if key in ('mag_gamma_act', 'ekf_gamma_act') else ''
        print(f"{key:<22}{obs:<24}{est:<10}"
              f"{c['acc']*100:>8.2f}%{c['keep']*100:>8.2f}%{marker}")
    print(f"{'Random (N='+str(rand_stats['n'])+')':<22}{'--':<24}{'--':<10}"
          f"{rand_stats['mean']*100:>5.2f}±{rand_stats['std']*100:.2f}%")
    print("=" * 78)

    # --- 两个核心 delta ---
    d_A = (results['mag_gamma_act']['acc'] - results['mag_gamma']['acc']) * 100
    d_B = (results['ekf_gamma_act']['acc'] - results['mag_gamma_act']['acc']) * 100
    d_vs_rand_mag = (results['mag_gamma_act']['acc'] - rand_stats['mean']) * 100
    d_vs_rand_ekf = (results['ekf_gamma_act']['acc'] - rand_stats['mean']) * 100
    z_mag_combo = (results['mag_gamma_act']['acc'] - rand_stats['mean']) / max(rand_stats['std'], 1e-8)
    z_ekf_combo = (results['ekf_gamma_act']['acc'] - rand_stats['mean']) / max(rand_stats['std'], 1e-8)

    print("\n=== TWO KEY COMPARISONS ===")
    print(f"  (A) Δ Mag-(γ×act) − Mag-γ            = {d_A:>+6.2f}%     "
          f"(组合观测避免 activation-only 崩溃 ?)")
    print(f"  (B) Δ EKF-(γ×act) − Mag-(γ×act)      = {d_B:>+6.2f}%     "
          f"★ EKF 获得非零增量 ?")
    print(f"      Δ Mag-(γ×act) − Rand.mean         = {d_vs_rand_mag:>+6.2f}%  "
          f"(z = {z_mag_combo:+.2f})")
    print(f"      Δ EKF-(γ×act) − Rand.mean         = {d_vs_rand_ekf:>+6.2f}%  "
          f"(z = {z_ekf_combo:+.2f})")

    results['key_deltas'] = {
        'A_mag_combo_vs_mag_gamma':       d_A,
        'B_ekf_combo_vs_mag_combo':       d_B,
        'mag_combo_vs_rand':              d_vs_rand_mag,
        'ekf_combo_vs_rand':              d_vs_rand_ekf,
        'z_mag_combo':                    float(z_mag_combo),
        'z_ekf_combo':                    float(z_ekf_combo),
    }

    # --- Verdict (按用户拍板的三档判据) ---
    print("\n=== VERDICT (formulation salvage) ===")

    # 情况 1: salvage 失败 (组合观测仍然明显差于 Mag-γ 或输给 Random)
    if d_A <= -2.0 or d_vs_rand_mag < 0:
        verdict = 'SALVAGE_FAILED'
        print(f"  [SALVAGE FAILED]")
        print(f"    (A) = {d_A:+.2f}% (组合观测比 Mag-γ 至少低 2% 或输给 Random)")
        print(f"    => dynamic observation 在 BN-ResNet 上无论怎么组合都无法救活 EKF")
        print(f"    => 结论: 直接走 Z. side project 叙事重构为")
        print(f"            'uncertainty-driven input-adaptive structured sparsification")
        print(f"             with BN-γ-based magnitude ranking'")
        print(f"    => EKF 保持专属主 SCI 论文的飞控场景, 不做方法通用性声明")
    # 情况 3: EKF 在有意义动态观测下终于获得增益
    elif d_A >= -0.5 and d_B >= 0.8:
        verdict = 'EKF_ALIVE'
        print(f"  [EKF ALIVE]  组合观测成功, 且 EKF 提供真实增量")
        print(f"    (A) = {d_A:+.2f}% (Mag-(γ×act) 接近或超过 Mag-γ)")
        print(f"    (B) = {d_B:+.2f}% ≥ +0.8% (EKF 贡献非零)")
        print(f"    => 'EKF-guided Input-Adaptive' 叙事终于成立")
        print(f"    => 下一步: Q/R grid 扫描 + trade-off curve")
    # 情况 2: 组合观测可行但 EKF 仍无用
    else:
        verdict = 'OBS_OK_EKF_USELESS'
        print(f"  [DYNAMIC OK, EKF STILL USELESS]")
        print(f"    (A) = {d_A:+.2f}% (组合观测接近/达到 Mag-γ, 避免了崩盘)")
        print(f"    (B) = {d_B:+.2f}% < +0.8% (EKF 仍未提供可识别增量)")
        print(f"    => dynamic obs 可以无害引入, 但 EKF 在本场景无用")
        print(f"    => 结论仍建议走 Z, 但增加一条诊断:")
        print(f"       'dynamic observation can be introduced without collapsing")
        print(f"        the ranking structure, but EKF still provides negligible")
        print(f"        incremental benefit under the current formulation.'")
    results['verdict'] = verdict

    print("=" * 78)

    # --- Plot ---
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10.5, 5.8))

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

        styles = [
            ('mag_gamma',     '#2ca02c', '-',  1.5, 'Mag-γ (v4)'),
            ('mag_act',       '#ff7f0e', ':',  1.3, 'Mag-act (v5 崩)'),
            ('mag_gamma_act', '#9467bd', '-',  2.2, 'Mag-(γ×act) ★'),
            ('ekf_gamma_act', '#d62728', '-',  2.6, 'EKF-(γ×act) ★'),
        ]
        for key, color, ls, lw, label in styles:
            acc = results[key]['acc'] * 100
            ax.axhline(acc, color=color, linestyle=ls, linewidth=lw,
                       label=f"{label} = {acc:.2f}%")

        ax.axhline(results['no_prune']['acc']*100, color='k',
                   linewidth=0.8, linestyle='-', alpha=0.3,
                   label=f"No prune = {results['no_prune']['acc']*100:.2f}%")

        ax.set_xlabel('Random seed index')
        ax.set_ylabel('Test accuracy (%)')
        title = (f"v5.5 Formulation Salvage (γ × activation composite obs)\n"
                 f"(A) Mag-combo − Mag-γ = {d_A:+.2f}%    "
                 f"(B) EKF-combo − Mag-combo = {d_B:+.2f}%    "
                 f"verdict: {verdict}")
        ax.set_title(title, fontsize=10)
        ax.legend(loc='lower right', fontsize=8, ncol=2, framealpha=0.92)
        ax.grid(alpha=0.3, zorder=1)
        ax.set_xticks(xs_r)
        ax.set_xlim(-0.6, len(rand_accs)-0.4)

        plt.tight_layout()
        png = os.path.join(args.out_dir, 'v5_5_salvage.png')
        plt.savefig(png, dpi=140)
        plt.close()
        print(f"\n[plot] saved to {png}")
    except Exception as e:
        print(f"\n[plot] skipped: {type(e).__name__}: {e}")

    out = os.path.join(args.out_dir, 'v5_5_salvage.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"[save] {out}")


if __name__ == '__main__':
    main()
