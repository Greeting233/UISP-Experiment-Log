"""
src/inference_ablation.py   [v2: fixed checkpoint loader]
================================================================================
Ablation: EKF vs Magnitude (|γ|) vs Random channel gating on ResNet-18 CIFAR-10.

v2 修复:
  - checkpoint loader 现在优先探测 'model_state_dict' (PyTorch 训练脚本最常见格式),
    而不是 fallback 到把整个 ckpt dict 塞给 load_state_dict 导致权重加载失败.
  - sanity check 失败时给更醒目的 error + 诊断建议.

Usage:
    %cd /content/drive/MyDrive/ekf_adaptive_pruning
    !python src/inference_ablation.py
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
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# ================================================================
# Gate 实现
# ================================================================

class _BaseGate(nn.Module):
    """公共基类. 子类实现 _compute_theta(z)."""
    def __init__(self, bn_layer, num_channels,
                 q_min=0.1, q_max=0.3, alpha=3.0,
                 warmup_steps=5, entropy_norm=None):
        super().__init__()
        self.bn_layer = bn_layer
        self.num_channels = num_channels
        self.q_min = q_min
        self.q_max = q_max
        self.alpha = alpha
        self.warmup_steps = warmup_steps
        self.entropy_norm = entropy_norm if entropy_norm is not None else math.log(10.0)

        self.register_buffer('step_count', torch.zeros(1, dtype=torch.long))
        self._logits_prev = None
        self.last_keep_ratio = 1.0

    def set_context(self, logits_prev):
        self._logits_prev = logits_prev

    def _uncertainty(self):
        if self._logits_prev is None:
            return 0.5
        p = F.softmax(self._logits_prev, dim=-1)
        ent = -(p * torch.log(p + 1e-8)).sum(-1).mean()
        return float(torch.clamp(ent / self.entropy_norm, 0.0, 1.0).item())

    def _compute_theta(self, z):
        raise NotImplementedError

    def _read_observation(self):
        return self.bn_layer.weight.detach().abs()

    def forward(self, x):
        self.step_count += 1
        z = self._read_observation()
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


class MagnitudeGate(_BaseGate):
    def _compute_theta(self, z):
        return z


class RandomGate(nn.Module):
    def __init__(self, bn_layer, num_channels,
                 q_min=0.1, q_max=0.3, alpha=3.0,
                 warmup_steps=5, entropy_norm=None, seed=42):
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
# Wrapper
# ================================================================

class GatedResNet(nn.Module):
    def __init__(self, base_model, gate_class, **gate_kwargs):
        super().__init__()
        self.base = base_model
        self.gates = nn.ModuleList()
        self._logits_prev = None

        for layer_name in ['layer3', 'layer4']:
            layer = getattr(self.base, layer_name)
            for block in layer:
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


# ================================================================
# Model loading [v2 fixed]
# ================================================================

def _build_builtin_resnet18_cifar(num_classes=10):
    class BasicBlock(nn.Module):
        expansion = 1
        def __init__(self, in_c, out_c, stride=1):
            super().__init__()
            self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_c)
            self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_c)
            self.shortcut = nn.Sequential()
            if stride != 1 or in_c != out_c:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_c, out_c, 1, stride, bias=False),
                    nn.BatchNorm2d(out_c)
                )
        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out = out + self.shortcut(x)
            return F.relu(out)

    class ResNet18CIFAR(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.layer1 = self._make(64, 64, 2, 1)
            self.layer2 = self._make(64, 128, 2, 2)
            self.layer3 = self._make(128, 256, 2, 2)
            self.layer4 = self._make(256, 512, 2, 2)
            self.fc = nn.Linear(512, num_classes)

        def _make(self, in_c, out_c, n, stride):
            blocks = [BasicBlock(in_c, out_c, stride)]
            for _ in range(n - 1):
                blocks.append(BasicBlock(out_c, out_c, 1))
            return nn.Sequential(*blocks)

        def forward(self, x):
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.layer1(x); x = self.layer2(x)
            x = self.layer3(x); x = self.layer4(x)
            x = F.adaptive_avg_pool2d(x, 1).flatten(1)
            return self.fc(x)

    return ResNet18CIFAR(num_classes=num_classes)


def _extract_state_dict(ckpt):
    """
    从各种常见 checkpoint 格式里提取真正的 state_dict.
    
    常见格式:
      - {'model_state_dict': {...}, 'epoch': ..., 'optimizer_state_dict': ...}  # 最常见
      - {'state_dict': {...}}                                                    # 次常见
      - {'model': {...}}, {'net': {...}}
      - 直接就是 state_dict
    """
    if not isinstance(ckpt, dict):
        return ckpt, 'direct'

    # 按优先级探测 wrapper key
    WRAPPER_KEYS = ['model_state_dict', 'state_dict', 'model', 'net', 'weights']
    for key in WRAPPER_KEYS:
        if key in ckpt and isinstance(ckpt[key], dict):
            sd = ckpt[key]
            sample = next(iter(sd.keys()), '')
            # state_dict 的 key 应该像 'conv1.weight' 这种形式
            if '.' in sample or sample.endswith(
                ('weight', 'bias', 'running_mean', 'running_var')):
                return sd, key

    # 没找到 wrapper, 检查 ckpt 本身是否就是 state_dict
    sample = next(iter(ckpt.keys()), '')
    if '.' in sample or sample.endswith(
        ('weight', 'bias', 'running_mean', 'running_var')):
        return ckpt, 'self'

    return ckpt, 'unknown'


def load_base_model(device, checkpoint_path):
    model = None
    try:
        from models import resnet18_cifar as m
        for name in ['resnet18_cifar', 'build_resnet18_cifar',
                     'ResNet18CIFAR', 'ResNet18', 'resnet18']:
            if hasattr(m, name):
                obj = getattr(m, name)
                try:
                    model = obj(num_classes=10)
                except TypeError:
                    model = obj()
                print(f"[model] using user's models.resnet18_cifar.{name}")
                break
    except ImportError as e:
        print(f"[model] import user's module failed: {e}")

    if model is None:
        print("[model] fallback to built-in ResNet18-CIFAR")
        model = _build_builtin_resnet18_cifar(num_classes=10)

    model = model.to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    sd, source = _extract_state_dict(ckpt)
    print(f"[ckpt] extracted state_dict from: '{source}'")
    print(f"[ckpt] total params in state_dict: {len(sd)}")

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[ckpt] WARNING: {len(missing)} missing keys, e.g. {missing[:3]}")
    if unexpected:
        print(f"[ckpt] WARNING: {len(unexpected)} unexpected keys, e.g. {unexpected[:3]}")
    if len(missing) > 5:
        print(f"[ckpt] !!!! state_dict SEVERELY mismatched !!!!")

    model.eval()
    return model


# ================================================================
# Data + Eval
# ================================================================

def get_test_loader(data_root='./data', batch_size=128, num_workers=2):
    tfm = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    ds = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=tfm
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


@torch.no_grad()
def evaluate(model, loader, device, track_keep=True):
    model.eval()
    correct, total = 0, 0
    keeps = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        pred = logits.argmax(-1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        if track_keep and hasattr(model, 'get_keep_ratios'):
            keeps.append(model.get_keep_ratios())

    acc = correct / total
    if keeps:
        keeps = np.array(keeps)
        per_gate_keep = keeps.mean(axis=0)
        avg_keep = float(per_gate_keep.mean())
    else:
        per_gate_keep, avg_keep = None, None
    return acc, avg_keep, per_gate_keep


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

    # --- 1/4: No pruning ---
    print("\n=== [1/4] No pruning (baseline sanity check) ===")
    acc, _, _ = evaluate(base, loader, device, track_keep=False)
    results['no_prune'] = {'acc': acc, 'keep': 1.0}
    print(f"  acc = {acc*100:.2f}%")
    if acc < 0.90:
        print("\n  [FATAL] baseline acc < 90%. checkpoint 加载仍然失败, 消融无意义.")
        print("  诊断命令 (在 Colab cell 里跑):")
        print("    import torch")
        print(f"    ckpt = torch.load('{args.ckpt}', map_location='cpu')")
        print("    print(type(ckpt))")
        print("    if isinstance(ckpt, dict): print(list(ckpt.keys()))")
        print("    # 然后把 keys 发给 Claude 定位问题")
        return

    # --- 2/4: EKF ---
    print("\n=== [2/4] EKF gating ===")
    m_ekf = GatedResNet(
        copy.deepcopy(base), EKFGate,
        Q=args.Q, R=args.R, **common_kwargs
    ).to(device)
    acc, keep, per = evaluate(m_ekf, loader, device)
    results['ekf'] = {'acc': acc, 'keep': keep, 'per_gate_keep': per.tolist()}
    print(f"  acc = {acc*100:.2f}%, avg keep = {keep:.4f}")
    print(f"  per-gate keep = {np.round(per, 4).tolist()}")

    # --- 3/4: Magnitude ---
    print("\n=== [3/4] Magnitude |γ| gating (核心消融) ===")
    m_mag = GatedResNet(
        copy.deepcopy(base), MagnitudeGate, **common_kwargs
    ).to(device)
    acc, keep, per = evaluate(m_mag, loader, device)
    results['magnitude'] = {'acc': acc, 'keep': keep, 'per_gate_keep': per.tolist()}
    print(f"  acc = {acc*100:.2f}%, avg keep = {keep:.4f}")
    print(f"  per-gate keep = {np.round(per, 4).tolist()}")

    # --- 4/4: Random ---
    print("\n=== [4/4] Random gating (matched keep ratio) ===")
    m_rand = GatedResNet(
        copy.deepcopy(base), RandomGate,
        seed=SEED, **common_kwargs
    ).to(device)
    acc, keep, per = evaluate(m_rand, loader, device)
    results['random'] = {'acc': acc, 'keep': keep, 'per_gate_keep': per.tolist()}
    print(f"  acc = {acc*100:.2f}%, avg keep = {keep:.4f}")
    print(f"  per-gate keep = {np.round(per, 4).tolist()}")

    # --- Summary ---
    print("\n" + "=" * 66)
    print("                      ABLATION SUMMARY")
    print("=" * 66)
    print(f"{'Method':<24} {'Acc':>10} {'Keep':>10}")
    print("-" * 66)
    print(f"{'No pruning':<24} {results['no_prune']['acc']*100:>9.2f}% {'100.00%':>10}")
    print(f"{'EKF (BN γ)':<24} {results['ekf']['acc']*100:>9.2f}% "
          f"{results['ekf']['keep']*100:>9.2f}%")
    print(f"{'Magnitude (|γ|)':<24} {results['magnitude']['acc']*100:>9.2f}% "
          f"{results['magnitude']['keep']*100:>9.2f}%")
    print(f"{'Random (matched)':<24} {results['random']['acc']*100:>9.2f}% "
          f"{results['random']['keep']*100:>9.2f}%")
    print("-" * 66)
    d_em = (results['ekf']['acc'] - results['magnitude']['acc']) * 100
    d_er = (results['ekf']['acc'] - results['random']['acc']) * 100
    d_mr = (results['magnitude']['acc'] - results['random']['acc']) * 100
    print(f"{'Δ EKF − Magnitude':<24} {d_em:>+9.2f}%     ← CORE ablation signal")
    print(f"{'Δ EKF − Random':<24} {d_er:>+9.2f}%     (v4 reference: +3.34%)")
    print(f"{'Δ Magnitude − Random':<24} {d_mr:>+9.2f}%     (quantile+uncertainty 贡献)")
    print("=" * 66)

    # --- 解读 ---
    print("\n=== Interpretation ===")
    if abs(d_em) < 0.5:
        print(f"  |Δ EKF-Mag| = {abs(d_em):.2f}% < 0.5%")
        print(f"  => EKF 对 BN γ 这种近似恒定观测几乎无增量价值")
        print(f"  => v4 的 Δ EKF-Random = {d_er:+.2f}% 实际来自 quantile+uncertainty 机制")
        print(f"  => 下一步: 切换到 activation-based 观测, 让 EKF 追踪 input-varying 信号")
    elif d_em >= 0.5:
        print(f"  Δ EKF-Mag = {d_em:+.2f}% ≥ 0.5%")
        print(f"  => EKF 时序平滑/协方差信息确有贡献, 核心方法站得住")
        print(f"  => 下一步: 扫 (q_min, q_max) 得 trade-off 曲线, 三方法同图对比")
    else:
        print(f"  Δ EKF-Mag = {d_em:+.2f}% (EKF 反向落后 Magnitude)")
        print(f"  => 可能 Q/R 超参过度平滑, 试 --Q 1e-2 --R 1e-3 对比")

    out_path = os.path.join(args.out_dir, 'ablation_v1.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[save] {out_path}")


if __name__ == '__main__':
    main()
