# src/analysis/summarize_results.py
"""
Phase 3 — 주 분석 결과 집계.

입력: evaluate_paper.py가 저장한 preds/pred_{mode}_seed{seed}_{split}.npz
출력:
  표 5.1  모델별 AP·위험재현율·MAE (5 seed 평균 ± 표준편차)
  표 5.2  사전등록 비교 (장면 단위 bootstrap 1000회 짝지은 Δ)
          전체−dummy, 속도만−dummy, 전체−속도만 의 ΔAP·95%CI·P(Δ>0)·판정
  5.2절   모델별 seed AP 범위

사용: python src/analysis/summarize_results.py --split test
"""
import argparse
import os
import numpy as np
from sklearn.metrics import average_precision_score

MODES = ['image_only', 'dummy', 'velocity_only', 'full']
DANGER = 3.0


def load(mode, seed, split, pdir):
    f = os.path.join(pdir, f'pred_{mode}_seed{seed}_{split}.npz')
    d = np.load(f, allow_pickle=True)
    return d['preds'], d['labels'], d['scenes']


def ap_of(preds, labels):
    bl = (labels < DANGER).astype(int)
    if bl.sum() == 0:
        return float('nan')
    return average_precision_score(bl, -preds)


def recall_of(preds, labels):
    pos = labels < DANGER
    if pos.sum() == 0:
        return float('nan')
    return float(((preds < DANGER) & pos).sum() / pos.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='test')
    ap.add_argument('--seeds', nargs='+', type=int,
                    default=[42, 123, 7, 555, 999])
    ap.add_argument('--pred_dir', default='preds')
    ap.add_argument('--n_boot', type=int, default=1000)
    args = ap.parse_args()

    # ── 로드 (모든 모델/seed는 동일한 프레임·씬 순서) ──
    preds = {m: [] for m in MODES}
    labels = scenes = None
    for m in MODES:
        for s in args.seeds:
            p, l, sc = load(m, s, args.split, args.pred_dir)
            preds[m].append(p)
            if labels is None:
                labels, scenes = l, sc
    for m in MODES:
        preds[m] = np.stack(preds[m])   # (n_seed, n_frame)

    bl = (labels < DANGER).astype(int)
    print(f"[{args.split}] 프레임 {len(labels)}, 위험 {int(bl.sum())} "
          f"({100*bl.mean():.2f}%)\n")

    # ── 표 5.1 : seed 평균 ± 표준편차 ──
    print("===== 표 5.1  모델별 성능 (5 seed 평균 ± 표준편차) =====")
    print(f"{'모델':14} {'AP':>16} {'위험재현율':>16} {'MAE':>14}")
    for m in MODES:
        aps = [ap_of(preds[m][i], labels) for i in range(len(args.seeds))]
        rec = [recall_of(preds[m][i], labels) for i in range(len(args.seeds))]
        mae = [float(np.abs(preds[m][i] - labels).mean())
               for i in range(len(args.seeds))]
        print(f"{m:14} {np.mean(aps):.4f}±{np.std(aps):.4f}   "
              f"{np.mean(rec):.4f}±{np.std(rec):.4f}   "
              f"{np.mean(mae):.4f}±{np.std(mae):.4f}")
    print()

    # ── 5.2절 : seed AP 범위 ──
    print("===== 5.2절  seed 간 AP 범위 =====")
    for m in MODES:
        aps = [ap_of(preds[m][i], labels) for i in range(len(args.seeds))]
        print(f"  {m:14}: {min(aps):.4f} ~ {max(aps):.4f}")
    print()

    # ── 표 5.2 : 장면 단위 bootstrap 짝지은 Δ ──
    uniq = sorted(set(scenes.tolist()))
    scene_to_idx = {s: np.where(scenes == s)[0] for s in uniq}
    rng = np.random.default_rng(42)

    def model_ap_on(idx, m):
        """resample 인덱스에서 모델 m의 seed 평균 AP."""
        lab = labels[idx]
        blb = (lab < DANGER).astype(int)
        if blb.sum() == 0:
            return np.nan
        vals = []
        for i in range(len(args.seeds)):
            vals.append(average_precision_score(blb, -preds[m][i][idx]))
        return float(np.mean(vals))

    comparisons = [('full', 'dummy'), ('velocity_only', 'dummy'),
                   ('full', 'velocity_only')]
    deltas = {c: [] for c in comparisons}

    for _ in range(args.n_boot):
        samp = rng.choice(len(uniq), size=len(uniq), replace=True)
        idx = np.concatenate([scene_to_idx[uniq[j]] for j in samp])
        ap_cache = {m: model_ap_on(idx, m) for m in
                    ('full', 'dummy', 'velocity_only')}
        for a, b in comparisons:
            deltas[(a, b)].append(ap_cache[a] - ap_cache[b])

    print("===== 표 5.2  사전등록 비교 (bootstrap %d회) =====" % args.n_boot)
    print(f"{'비교':24} {'ΔAP':>9} {'95% CI':>22} {'P(Δ>0)':>8}  판정")
    for a, b in comparisons:
        d = np.array(deltas[(a, b)])
        d = d[~np.isnan(d)]
        lo, hi = np.percentile(d, [2.5, 97.5])
        p_pos = float((d > 0).mean())
        verdict = "구분됨" if (lo > 0 or hi < 0) else "구분 불가"
        print(f"{a+'−'+b:24} {d.mean():>+9.4f} "
              f"[{lo:>+8.4f},{hi:>+8.4f}] {p_pos:>8.3f}  {verdict}")
    print()


if __name__ == '__main__':
    main()