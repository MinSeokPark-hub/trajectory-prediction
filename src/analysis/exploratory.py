# src/analysis/exploratory.py
"""
Phase 5 — 탐색적 분석 (논문 표 5.3).

구간(직진/회전) × 위험기준(2·3·4초)별로 전체ego−dummy 의 ΔAP를
장면 단위 bootstrap으로 계산한다. (사전등록 아님 → 가설 생성용)

  - 직진 구간, 3초
  - 전체, 4초 기준
  - 전체, 2초 기준 (결론 사용 안 함)
  - 회전 구간 (장면 수 적음 → 해석 보류, 수치만)

직진/회전 구분: 마지막 프레임 |회전속도| > 0.08 rad/s 이면 회전.  논문 3.3

CTRV 대안 라벨 D(표 5.3의 'C·D 합의' 행)는 compute_label_d.py로 따로 생성.

사용: python src/analysis/exploratory.py --split test
"""
import argparse
import os
import numpy as np
from sklearn.metrics import average_precision_score

SEEDS_DEFAULT = [42, 123, 7, 555, 999]
ROT_RATE_IDX = 17
TURN_THRESH = 0.08


def load_preds(mode, seeds, split, pdir):
    arr = []
    labels = scenes = None
    for s in seeds:
        d = np.load(os.path.join(pdir, f'pred_{mode}_seed{s}_{split}.npz'),
                    allow_pickle=True)
        arr.append(d['preds'])
        labels, scenes = d['labels'], d['scenes']
    return np.stack(arr), labels, scenes


def bootstrap_delta(sub, labels, scenes, preds_a, preds_b, thr,
                    n_boot=1000, seed=42):
    """sub(인덱스 부분집합)에서 A−B 의 ΔAP bootstrap."""
    scenes_sub = scenes[sub]
    uniq = sorted(set(scenes_sub.tolist()))
    s2i = {s: sub[np.where(scenes_sub == s)[0]] for s in uniq}
    rng = np.random.default_rng(seed)
    n_seed = preds_a.shape[0]

    def mean_ap(idx, preds):
        lab = labels[idx]
        bl = (lab < thr).astype(int)
        if bl.sum() == 0:
            return np.nan
        return float(np.mean([average_precision_score(bl, -preds[i][idx])
                              for i in range(n_seed)]))

    deltas = []
    for _ in range(n_boot):
        samp = rng.choice(len(uniq), size=len(uniq), replace=True)
        idx = np.concatenate([s2i[uniq[j]] for j in samp])
        da = mean_ap(idx, preds_a)
        db = mean_ap(idx, preds_b)
        deltas.append(da - db)
    d = np.array(deltas)
    d = d[~np.isnan(d)]
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi), float((d > 0).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='test')
    ap.add_argument('--seeds', nargs='+', type=int, default=SEEDS_DEFAULT)
    ap.add_argument('--pred_dir', default='preds')
    ap.add_argument('--cache_dir', default='cache')
    ap.add_argument('--n_boot', type=int, default=1000)
    args = ap.parse_args()

    full, labels, scenes = load_preds('full', args.seeds, args.split, args.pred_dir)
    dummy, _, _ = load_preds('dummy', args.seeds, args.split, args.pred_dir)

    ego = np.load(os.path.join(args.cache_dir, f'frames_{args.split}.npz'),
                  allow_pickle=True)['ego_raw']
    turn_mask = np.abs(ego[:, ROT_RATE_IDX]) > TURN_THRESH
    straight = np.where(~turn_mask)[0]
    turning  = np.where(turn_mask)[0]
    allidx   = np.arange(len(labels))

    def risk_count(idx, thr):
        return int((labels[idx] < thr).sum()), len(set(scenes[idx].tolist()))

    print(f"[{args.split}] 전체 {len(labels)} | 직진 {len(straight)} "
          f"| 회전 {len(turning)}\n")
    print("===== 표 5.3  탐색적 ΔAP (전체 ego − dummy) =====")
    print(f"{'조건':22} {'위험/장면':>12} {'ΔAP':>9} "
          f"{'95% CI':>22} {'P(Δ>0)':>8}")

    rows = [
        ("직진, 3초", straight, 3.0),
        ("전체, 4초", allidx, 4.0),
        ("전체, 2초", allidx, 2.0),
        ("회전(해석보류)", turning, 3.0),
    ]
    for name, idx, thr in rows:
        rc, nsc = risk_count(idx, thr)
        if rc == 0:
            print(f"{name:22} {f'{rc}/{nsc}':>12}  위험 프레임 없음")
            continue
        dmean, lo, hi, p = bootstrap_delta(idx, labels, scenes, full, dummy,
                                           thr, args.n_boot)
        note = "  (결론 미사용)" if "2초" in name else (
               "  (해석 보류)" if "회전" in name else "")
        print(f"{name:22} {f'{rc}/{nsc}':>12} {dmean:>+9.4f} "
              f"[{lo:>+8.4f},{hi:>+8.4f}] {p:>8.3f}{note}")
    print()


if __name__ == '__main__':
    main()