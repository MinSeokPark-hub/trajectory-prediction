# src/analysis/baselines.py
"""
Phase 4 — 기준선 (논문 표 5.1 윗부분, 3.5절).

1. 무작위 순위       : AP = 위험 비율
2. 상수 예측기(10초) : 항상 TTC=10 → MAE, 위험재현율(=0)
3. 회전 속도 크기만  : 마지막 프레임 |회전속도|로 순위 → AP
4. ego 20차원 GBDT   : HistGradientBoostingClassifier(train) → test AP

입력: cache/frames_{train,test}.npz (ego_raw, labels)
사용: python src/analysis/baselines.py --split test
"""
import argparse
import os
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.ensemble import HistGradientBoostingClassifier

DANGER = 3.0
ROT_RATE_IDX = 17   # 마지막(현재) 프레임의 회전속도 (4번째 블록, offset 2)


def load_cache(split, cache_dir):
    d = np.load(os.path.join(cache_dir, f'frames_{split}.npz'),
                allow_pickle=True)
    return d['ego_raw'].astype(np.float32), d['labels'].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='test')
    ap.add_argument('--cache_dir', default='cache')
    args = ap.parse_args()

    ego_tr, lab_tr = load_cache('train', args.cache_dir)
    ego_te, lab_te = load_cache(args.split, args.cache_dir)

    bl_te = (lab_te < DANGER).astype(int)
    risk_ratio = float(bl_te.mean())

    print(f"[{args.split}] 프레임 {len(lab_te)}, 위험 {int(bl_te.sum())} "
          f"({100*risk_ratio:.2f}%)\n")
    print("===== 기준선 =====")

    # 1) 무작위
    print(f"  무작위 순위        AP = {risk_ratio:.4f} (= 위험 비율)")

    # 2) 상수 예측기 (10초)
    const_pred = np.full_like(lab_te, 10.0)
    mae_c = float(np.abs(const_pred - lab_te).mean())
    rec_c = float(((const_pred < DANGER) & (lab_te < DANGER)).sum()
                  / max((lab_te < DANGER).sum(), 1))
    print(f"  상수예측기(10초)   MAE = {mae_c:.4f},  위험재현율 = {rec_c:.4f}")

    # 3) 회전 속도 크기만
    rot_score = np.abs(ego_te[:, ROT_RATE_IDX])
    ap_rot = average_precision_score(bl_te, rot_score)
    print(f"  회전속도 크기만    AP = {ap_rot:.4f}")

    # 4) ego 20차원 GBDT
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05,
                                         random_state=42)
    clf.fit(ego_tr, (lab_tr < DANGER).astype(int))
    proba = clf.predict_proba(ego_te)[:, 1]
    ap_gbdt = average_precision_score(bl_te, proba)
    print(f"  ego-motion GBDT    AP = {ap_gbdt:.4f}")
    print()


if __name__ == '__main__':
    main()