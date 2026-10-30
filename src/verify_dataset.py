# src/verify_dataset.py
"""
프레임 단위 데이터셋을 빌드·캐시하고, 프레임/위험 프레임 수가
논문 표 3.1과 맞는지 확인한다.

논문 표 3.1 (목표치)
  split   frames   risk   risk%
  train   21,515   409    1.90%
  val      4,599   111    2.41%
  test     4,628   137    2.96%
  합계    30,742   657    2.14%

사용: python src/verify_dataset.py
"""
import os
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.nuscenes_ttc_dataset import NuScenesTTCDataset, get_scene_splits

TARGET = {
    'train': (21515, 409),
    'val':   (4599, 111),
    'test':  (4628, 137),
}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataroot', default='/workspace/data/nuscenes')
    ap.add_argument('--version', default='v1.0-trainval')
    ap.add_argument('--cache_dir', default='cache')
    args = ap.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    from nuscenes.nuscenes import NuScenes
    print(f"[NuScenes] Loading {args.version} ...")
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    print(f"\n{'split':6} {'frames':>8} {'(target)':>10} "
          f"{'risk':>6} {'(target)':>10} {'risk%':>7}")
    print("-" * 56)

    total_f = total_r = 0
    for split in ['train', 'val', 'test']:
        cache = os.path.join(args.cache_dir, f'frames_{split}.npz')
        ds = NuScenesTTCDataset(nusc, split=split, cache_path=cache)
        n = len(ds)
        r = int((ds.labels < 3.0).sum())
        tf, tr = TARGET[split]
        total_f += n; total_r += r
        print(f"{split:6} {n:>8} {tf:>10} {r:>6} {tr:>10} "
              f"{100*r/max(n,1):>6.2f}%")

    print("-" * 56)
    print(f"{'합계':6} {total_f:>8} {30742:>10} {total_r:>6} {657:>10} "
          f"{100*total_r/max(total_f,1):>6.2f}%")
    print("\n논문 표 3.1과 근접하면(±수십 프레임) 방법론 정합 성공.")


if __name__ == '__main__':
    main()