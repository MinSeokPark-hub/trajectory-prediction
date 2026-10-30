# src/train_paper.py
"""
논문 재현 학습 스크립트 (방법론 정합 버전).

논문과 맞춘 점
------------------------------------------------------------------
- 프레임 단위 최소 TTC 라벨 (Dataset에서 처리)
- ego 20차원 특징은 학습 평균/표준편차로 표준화 (train 통계를 val/test에 적용)
- 1 epoch에는 ResNet-18 본체 고정, 새 층만 학습 → 2 epoch부터 전체 학습
- 평가는 '마지막 epoch' 모델 사용 (best-val 선택은 편향, 논문 4.3)
- seed 재현성: --seed (논문은 42,123,7,555,999 5개)
- Weighted L1 (17.53 / 6.96 / 0.36), Adam lr=5e-5, batch=16, 3 epoch

사용 예:
  for S in 42 123 7 555 999; do
    python src/train_paper.py --mode full --seed $S
  done
"""
import argparse
import os
import sys
import time
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.nuscenes_ttc_dataset import NuScenesTTCDataset
from src.models.resnet_ttc import build_model

TTC_MAX = 10.0


# ── Weighted L1 Loss (논문 3.4) ───────────────────────────────
class WeightedL1Loss(nn.Module):
    def __init__(self, w_danger=17.53, w_warn=6.96, w_safe=0.36):
        super().__init__()
        self.w_danger, self.w_warn, self.w_safe = w_danger, w_warn, w_safe

    def forward(self, pred, target):
        l1 = torch.abs(pred - target)
        w = torch.where(target < 3.0, self.w_danger,
            torch.where(target < 5.0, self.w_warn, self.w_safe))
        return (w * l1).mean()


# ── 백본 freeze/unfreeze ──────────────────────────────────────
def set_backbone_frozen(model, frozen: bool):
    for p in model.image_encoder.parameters():
        p.requires_grad = not frozen


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── 학습/검증 루프 ────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total, t0 = 0.0, time.time()
    for step, (images, ego, labels) in enumerate(loader):
        images, ego, labels = images.to(device), ego.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images, ego), labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
        if step % 100 == 0:
            print(f"  E{epoch} Step {step}/{len(loader)}  "
                  f"Loss={loss.item():.4f}  {time.time()-t0:.0f}s")
    return total / len(loader)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total = 0.0
    for images, ego, labels in loader:
        images, ego, labels = images.to(device), ego.to(device), labels.to(device)
        total += criterion(model(images, ego), labels).item()
    return total / len(loader)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataroot', default='/workspace/data/nuscenes')
    ap.add_argument('--version', default='v1.0-trainval')
    ap.add_argument('--mode', default='full',
                    choices=['image_only', 'dummy', 'velocity_only', 'full'])
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--lr', type=float, default=5e-5)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--outdir', default='checkpoints')
    ap.add_argument('--cache_dir', default='cache')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    seed_everything(args.seed)

    from nuscenes.nuscenes import NuScenes
    print(f"[NuScenes] Loading {args.version} ...")
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    cache_train = os.path.join(args.cache_dir, 'frames_train.npz')
    cache_val   = os.path.join(args.cache_dir, 'frames_val.npz')
    stats_path  = os.path.join(args.cache_dir, 'ego_stats.npz')

    # ── 학습셋 (+ ego 표준화 통계) ──
    train_ds = NuScenesTTCDataset(nusc, split='train', cache_path=cache_train)
    if os.path.exists(stats_path):
        s = np.load(stats_path)
        train_ds.ego_mean, train_ds.ego_std = s['mean'], s['std']
    else:
        np.savez(stats_path, mean=train_ds.ego_mean, std=train_ds.ego_std)
        print(f"[Stats] ego 표준화 통계 저장: {stats_path}")
    print(f"  Train frames: {len(train_ds)}")

    # ── 검증셋 (학습 통계 적용) ──
    val_ds = NuScenesTTCDataset(nusc, split='val', cache_path=cache_val,
                                ego_stats=(train_ds.ego_mean, train_ds.ego_std))
    print(f"  Val frames: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=False)
    val_loader   = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                              num_workers=args.workers, pin_memory=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Train] Device {device}  Mode {args.mode}  Seed {args.seed}")

    model = build_model(args.mode).to(device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    criterion = WeightedL1Loss()

    tag = f"{args.mode}_seed{args.seed}"
    for epoch in range(1, args.epochs + 1):
        # 1 epoch: 백본 고정, 2 epoch부터 전체 학습
        if epoch == 1:
            set_backbone_frozen(model, True)
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        elif epoch == 2:
            set_backbone_frozen(model, False)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        tr = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        va = eval_epoch(model, val_loader, criterion, device)
        print(f"Epoch {epoch}  TrainLoss={tr:.4f}  ValLoss={va:.4f}")

        # 마지막 epoch 모델을 평가에 사용 → 매 epoch last 덮어쓰기
        last = os.path.join(args.outdir, f'last_{tag}.pt')
        torch.save({'epoch': epoch, 'mode': args.mode, 'seed': args.seed,
                    'state_dict': model.state_dict(), 'val_loss': va}, last)

    print(f"[Done] {tag}  (마지막 epoch 모델: {last})")


if __name__ == '__main__':
    main()