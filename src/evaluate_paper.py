# src/evaluate_paper.py
"""
논문 재현 평가 스크립트.
메트릭: AP (Average Precision), Danger Recall (TTC < 3s)
씬 단위 Bootstrap CI (1000회).

사용법:
  python src/evaluate_paper.py \
    --dataroot /workspace/data/nuscenes \
    --ckpt checkpoints/best_full.pt \
    --mode full
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

DANGER_THRESH = 3.0   # TTC < 3s → 위험
TTC_CAP       = 10.0


# ── AP 계산 ────────────────────────────────────────────────────
def compute_ap(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    scores  : 위험도 점수 (TTC를 역변환: 높을수록 위험)
    labels  : 1=위험(TTC<3), 0=안전
    """
    from sklearn.metrics import average_precision_score
    if labels.sum() == 0:
        return float('nan')
    return float(average_precision_score(labels, scores))


def compute_danger_recall(preds: np.ndarray, labels: np.ndarray,
                          threshold: float = DANGER_THRESH) -> float:
    """preds, labels 모두 TTC 값."""
    pos_mask = labels < threshold
    if pos_mask.sum() == 0:
        return float('nan')
    tp = ((preds < threshold) & pos_mask).sum()
    return float(tp / pos_mask.sum())


# ── Bootstrap CI ───────────────────────────────────────────────
def bootstrap_ci(values: np.ndarray, n_iter: int = 1000, ci: float = 0.95):
    """씬 단위 bootstrap confidence interval."""
    means = []
    n = len(values)
    rng = np.random.default_rng(42)
    for _ in range(n_iter):
        idx = rng.integers(0, n, size=n)
        means.append(values[idx].mean())
    means = np.array(means)
    lo = np.percentile(means, (1 - ci) / 2 * 100)
    hi = np.percentile(means, (1 + ci) / 2 * 100)
    return float(means.mean()), float(lo), float(hi)


# ── 추론 ──────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()
    all_preds  = []
    all_labels = []
    for images, ego, labels in loader:
        images = images.to(device)
        ego    = ego.to(device)
        pred   = model(images, ego).cpu().numpy()
        all_preds.append(pred)
        all_labels.append(labels.numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', default='/workspace/data/nuscenes')
    parser.add_argument('--version',  default='v1.0-trainval')
    parser.add_argument('--ckpt',     required=True)
    parser.add_argument('--mode',     default='full',
                        choices=['image_only', 'dummy', 'velocity_only', 'full'])
    parser.add_argument('--batch',    type=int, default=32)
    parser.add_argument('--workers',  type=int, default=4)
    parser.add_argument('--split',    default='test',
                        choices=['val', 'test'])
    parser.add_argument('--cache_dir', default='cache')
    args = parser.parse_args()

    from nuscenes.nuscenes import NuScenes
    print(f"[NuScenes] Loading {args.version}...")
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    cache_test = os.path.join(args.cache_dir,
                              f'labels_{args.split}_{args.version}.json')
    ds = NuScenesTTCDataset(nusc, split=args.split, cache_path=cache_test)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = build_model(args.mode).to(device)
    ckpt   = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    print(f"[Eval] Loaded {args.ckpt}  (mode={args.mode})")

    preds, labels = run_inference(model, loader, device)

    # 위험도 점수 (높을수록 위험) = TTC_CAP - pred
    scores  = TTC_CAP - np.clip(preds, 0, TTC_CAP)
    bin_lbl = (labels < DANGER_THRESH).astype(int)

    ap     = compute_ap(scores, bin_lbl)
    recall = compute_danger_recall(preds, labels)
    mae    = float(np.abs(preds - labels).mean())

    print("\n===== 평가 결과 =====")
    print(f"  AP            : {ap:.4f}")
    print(f"  Danger Recall : {recall:.4f}")
    print(f"  MAE           : {mae:.4f}")

    # 씬별 Bootstrap CI
    # — 씬 token을 sample_token 기준으로 매핑
    splits = get_scene_splits(nusc)
    scene_tokens = splits[args.split]
    sample_to_scene = {}
    for st in scene_tokens:
        scene = nusc.get('scene', st)
        tok = scene['first_sample_token']
        while tok:
            sample_to_scene[tok] = st
            tok = nusc.get('sample', tok)['next'] or None

    # 씬별 AP 수집
    scene_aps = {}
    for (sample_token, _), pred, label, score, bl in zip(
            ds.samples, preds, labels, scores, bin_lbl):
        sc = sample_to_scene.get(sample_token, 'unknown')
        if sc not in scene_aps:
            scene_aps[sc] = {'scores': [], 'labels': []}
        scene_aps[sc]['scores'].append(score)
        scene_aps[sc]['labels'].append(bl)

    per_scene_ap = []
    for sc, d in scene_aps.items():
        if sum(d['labels']) > 0:
            per_scene_ap.append(compute_ap(
                np.array(d['scores']), np.array(d['labels'])))

    if per_scene_ap:
        mean_ap, lo, hi = bootstrap_ci(np.array(per_scene_ap))
        print(f"\n  Bootstrap AP  : {mean_ap:.4f} "
              f"[95% CI: {lo:.4f} – {hi:.4f}]  (씬 수: {len(per_scene_ap)})")

    print("=====================\n")


if __name__ == '__main__':
    main()
