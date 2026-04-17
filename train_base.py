"""ResNet18 CIFAR-10 baseline training."""
import os
import argparse
import time
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

from models.resnet18_cifar import resnet18_cifar


def get_dataloaders(data_dir="./data", batch_size=128, num_workers=2):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    train_tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(mean, std)])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    train_set = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_tf)
    test_set = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_tf)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, n_batches = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--save_path", type=str, default="./checkpoints/resnet18_cifar_base.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    if device.type == "cuda":
        print(f"[gpu]    {torch.cuda.get_device_name(0)}")

    train_loader, test_loader = get_dataloaders(data_dir=args.data_dir, batch_size=args.batch_size)

    model = resnet18_cifar(num_classes=10).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model]  ResNet18-CIFAR, params={n_params/1e6:.2f}M")

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    best_acc = 0.0
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()
        test_acc = evaluate(model, test_loader, device)
        dt = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"[ep {epoch:3d}/{args.epochs}] loss={train_loss:.4f}  test_acc={test_acc:.2f}%  lr={lr_now:.4f}  dt={dt:.1f}s")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "test_acc": test_acc}, args.save_path)

    total_time = (time.time() - t_start) / 60
    print(f"\n[done] best_acc={best_acc:.2f}%  total_time={total_time:.1f}min  saved={args.save_path}")


if __name__ == "__main__":
    main()
