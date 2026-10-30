# src/analysis/linear_probe.py
"""
Phase 6a — 이미지 특징의 자차 운동 표현력 (논문 4.4).

학습된 모델의 이미지 특징(256차원)에서 자차 운동(속도·가속도·회전속도)을
선형회귀로 예측하고 결정계수 R²를 보고한다. 회전속도·가속도가 거의
설명되지 않으면 '자차 운동을 별도 입력으로' 주는 근거가 된다.

사용:
  python src/analysis/linear_probe.py --ckpt checkpoints/last_full_seed42.pt
"""
import argparse
import os
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.nuscenes_ttc_dataset import NuScenesTTCDataset
from src.models.resnet_ttc import build_model

# ego 20차원 중 현재(마지막) 프레임 인덱스
IDX = {'속도': 15, '종가속도': 16, '회전속도': 17}


@torch.no_grad()
def extract_feats(model, loader, device):
    model.eval()
    feats = []
    for images, ego, labels in loader:
        B, T, C, H, W = images.shape
        x = images.view(B * T, C, H, W).to(device)
        f = model.image_encoder(x).view(B, T, 512).mean(1)   # (B,512)
        f = model.image_proj(f)                               # (B,256)
        feats.append(f.cpu().numpy())
    return np.concatenate(feats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataroot', default='/workspace/data/nuscenes')
    ap.add_argument('--version', default='v1.0-trainval')
    ap.add_argument('--ckpt', default='checkpoints/last_full_seed42.pt')
    ap.add_argument('--split', default='train')
    ap.add_argument('--cache_dir', default='cache')
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()

    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    s = np.load(os.path.join(args.cache_dir, 'ego_stats.npz'))
    ds = NuScenesTTCDataset(nusc, split=args.split,
                            cache_path=os.path.join(args.cache_dir,
                                                    f'frames_{args.split}.npz'),
                            ego_stats=(s['mean'], s['std']))
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model('full').to(device)
    st = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(st['state_dict'])
    print(f"[Probe] 특징 추출: {args.ckpt}  ({args.split}, {len(ds)}프레임)")

    X = extract_feats(model, loader, device)      # (N,256)
    ego_raw = ds.ego_raw                          # (N,20) 표준화 전

    print("\n===== 이미지 특징 → 자차 운동 선형회귀 R² (5-fold CV) =====")
    for name, idx in IDX.items():
        y = ego_raw[:, idx]
        r2 = cross_val_score(LinearRegression(), X, y, cv=5,
                             scoring='r2').mean()
        print(f"  {name:8}: R² = {r2:.3f}")
    print("\n(회전속도·가속도가 낮으면 평균 구조가 시간순서를 잃는다는 근거)")


if __name__ == '__main__':
    main()