# src/train_paper.py
"""
논문 재현 학습 스크립트.
사용법:
  python src/train_paper.py --dataroot /workspace/data/nuscenes \
                            --mode full --epochs 3 --batch 16
"""
import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.nuscenes_ttc_dataset import NuScenesTTCDataset
from src.models.resnet_ttc import build_model

TTC_MAX = 10.0


# ── Weighted L1 Loss (논문 Table 1 기준) ──────────────────────
class WeightedL1Loss(nn.Module):
    def __init__(self, w_danger=17.53, w_warn=6.96, w_safe=0.36):
        super().__init__()
        self.w_danger = w_danger   # TTC < 3s
        self.w_warn   = w_warn     # 3 ≤ TTC < 5s
        self.w_safe   = w_safe     # TTC ≥ 5s

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        l1 = torch.abs(pred - target)
        w = torch.where(target < 3.0, self.w_danger,
            torch.where(target < 5.0, self.w_warn, self.w_safe))
        return (w * l1).mean()


# ── 학습 루프 ─────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0.0
    t0 = time.time()
    for step, (images, ego, labels) in enumerate(loader):
        images = images.to(device)
        ego    = ego.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        pred = model(images, ego)
        loss = criterion(pred, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        if step % 50 == 0:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch} Step {step}/{len(loader)}  "
                  f"Loss={loss.item():.4f}  Time={elapsed:.1f}s")

    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    for images, ego, labels in loader:
        images = images.to(device)
        ego    = ego.to(device)
        labels = labels.to(device)
        pred = model(images, ego)
        total_loss += criterion(pred, labels).item()
    return total_loss / len(loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', default='/workspace/data/nuscenes')
    parser.add_argument('--version', default='v1.0-trainval')
    parser.add_argument('--mode', default='full',
                        choices=['image_only', 'dummy', 'velocity_only', 'full'])
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--outdir', default='checkpoints')
    parser.add_argument('--cache_dir', default='cache')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    from nuscenes.nuscenes import NuScenes
    print(f"[NuScenes] Loading {args.version} from {args.dataroot}")
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    cache_train = os.path.join(args.cache_dir, f'labels_train_{args.version}.json')
    cache_val   = os.path.join(args.cache_dir, f'labels_val_{args.version}.json')

    print("[Dataset] Train set 구성 중...")
    train_ds = NuScenesTTCDataset(nusc, split='train', cache_path=cache_train)
    print(f"  Train samples: {len(train_ds)}")

    print("[Dataset] Val set 구성 중...")
    val_ds = NuScenesTTCDataset(nusc, split='val', cache_path=cache_val)
    print(f"  Val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=args.workers, pin_memory=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Train] Device: {device}  Mode: {args.mode}")

    model     = build_model(args.mode).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = WeightedL1Loss()

    best_val_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_loss   = eval_epoch(model, val_loader, criterion, device)
        print(f"Epoch {epoch}  TrainLoss={train_loss:.4f}  ValLoss={val_loss:.4f}")

        ckpt = os.path.join(args.outdir, f'resnet_ttc_{args.mode}_ep{epoch}.pt')
        torch.save({'epoch': epoch, 'mode': args.mode,
                    'state_dict': model.state_dict(),
                    'val_loss': val_loss}, ckpt)
        print(f"  Saved: {ckpt}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best = os.path.join(args.outdir, f'best_{args.mode}.pt')
            torch.save({'epoch': epoch, 'mode': args.mode,
                        'state_dict': model.state_dict(),
                        'val_loss': val_loss}, best)
            print(f"  ★ Best updated: {best}")

    print(f"\n[Done] Best val loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    main()
