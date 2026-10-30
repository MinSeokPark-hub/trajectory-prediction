# src/evaluate_paper.py
"""
단일 체크포인트 평가 (방법론 정합 버전).
지표: AP, 위험 재현율(TTC<3s), MAE.  논문 3.5

- ego 표준화 통계(cache/ego_stats.npz)를 학습과 동일하게 적용
- 예측 결과(preds/labels/프레임/씬)를 .npz로 저장 → Phase 3(부트스트랩 Δ),
  Phase 5(구간별 분석)에서 재사용

사용 예:
  python src/evaluate_paper.py --mode full --seed 42 --split test
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.nuscenes_ttc_dataset import NuScenesTTCDataset, get_scene_splits
from src.models.resnet_ttc import build_model

DANGER_THRESH = 3.0


def compute_ap(scores, labels_bin):
    from sklearn.metrics import average_precision_score
    if labels_bin.sum() == 0:
        return float('nan')
    return float(average_precision_score(labels_bin, scores))


def danger_recall(preds, labels, thr=DANGER_THRESH):
    pos = labels < thr
    if pos.sum() == 0:
        return float('nan')
    return float(((preds < thr) & pos).sum() / pos.sum())


@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()
    P, L = [], []
    for images, ego, labels in loader:
        images, ego = images.to(device), ego.to(device)
        P.append(model(images, ego).cpu().numpy())
        L.append(labels.numpy())
    return np.concatenate(P), np.concatenate(L)


def frame_to_scene(nusc, split):
    """각 프레임 sample_token → scene_token 매핑."""
    splits = get_scene_splits(nusc)
    m = {}
    for sc in splits[split]:
        tok = nusc.get('scene', sc)['first_sample_token']
        while tok:
            m[tok] = sc
            tok = nusc.get('sample', tok)['next']
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataroot', default='/workspace/data/nuscenes')
    ap.add_argument('--version', default='v1.0-trainval')
    ap.add_argument('--mode', default='full',
                    choices=['image_only', 'dummy', 'velocity_only', 'full'])
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--ckpt', default=None, help='없으면 last_{mode}_seed{seed}.pt')
    ap.add_argument('--split', default='test', choices=['val', 'test'])
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--outdir', default='checkpoints')
    ap.add_argument('--cache_dir', default='cache')
    ap.add_argument('--pred_dir', default='preds')
    args = ap.parse_args()

    from nuscenes.nuscenes import NuScenes
    print(f"[NuScenes] Loading {args.version} ...")
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    # ego 표준화 통계 (학습과 동일)
    stats_path = os.path.join(args.cache_dir, 'ego_stats.npz')
    s = np.load(stats_path)
    ego_stats = (s['mean'], s['std'])

    cache = os.path.join(args.cache_dir, f'frames_{args.split}.npz')
    ds = NuScenesTTCDataset(nusc, split=args.split, cache_path=cache,
                            ego_stats=ego_stats)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, pin_memory=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(args.mode).to(device)
    ckpt = args.ckpt or os.path.join(args.outdir,
                                     f'last_{args.mode}_seed{args.seed}.pt')
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state['state_dict'])
    print(f"[Eval] {ckpt}  mode={args.mode}  split={args.split}")

    preds, labels = run_inference(model, loader, device)

    scores  = -preds                              # 위험도 = -예측TTC (논문 3.5)
    bin_lbl = (labels < DANGER_THRESH).astype(int)

    ap_val = compute_ap(scores, bin_lbl)
    rec    = danger_recall(preds, labels)
    mae    = float(np.abs(preds - labels).mean())

    print("\n===== 평가 결과 =====")
    print(f"  Split         : {args.split}  (프레임 {len(labels)}, "
          f"위험 {int(bin_lbl.sum())})")
    print(f"  AP            : {ap_val:.4f}")
    print(f"  Danger Recall : {rec:.4f}")
    print(f"  MAE           : {mae:.4f}")
    print("=====================\n")

    # Phase 3/5용 예측 저장
    os.makedirs(args.pred_dir, exist_ok=True)
    f2s = frame_to_scene(nusc, args.split)
    scenes = np.array([f2s.get(t, 'NA') for t in ds.frames])
    out = os.path.join(args.pred_dir,
                       f'pred_{args.mode}_seed{args.seed}_{args.split}.npz')
    np.savez(out, preds=preds, labels=labels,
             frames=np.array(ds.frames, dtype=object), scenes=scenes)
    print(f"[Saved] {out}")


if __name__ == '__main__':
    main()