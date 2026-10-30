# src/analysis/cd_agreement_row.py
"""
표 5.3 'C·D 합의 위험, 직진' 행.

직진 구간에서, 위험 라벨을 'C(직선)·D(CTRV) 모두 위험'으로 더 엄격히 정의하고
(= 라벨 인공물이 작은 합의 프레임만 양성) 전체ego−dummy 의 ΔAP를
장면 단위 bootstrap으로 계산한다.

입력: preds/pred_{full,dummy}_seed*_test.npz, preds/label_D_test.npz,
      cache/frames_test.npz
사용: python src/analysis/cd_agreement_row.py
"""
import argparse, os
import numpy as np
from sklearn.metrics import average_precision_score

SEEDS = [42,123,7,555,999]
DANGER = 3.0; ROT_IDX = 17; TURN = 0.08


def load(mode, pdir):
    arr=[]; labels=scenes=frames=None
    for s in SEEDS:
        d=np.load(os.path.join(pdir,f'pred_{mode}_seed{s}_test.npz'),allow_pickle=True)
        arr.append(d['preds']); labels=d['labels']; scenes=d['scenes']; frames=d['frames']
    return np.stack(arr),labels,scenes,frames


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--pred_dir',default='preds'); ap.add_argument('--cache_dir',default='cache')
    ap.add_argument('--n_boot',type=int,default=1000)
    args=ap.parse_args()

    full,labels,scenes,frames=load('full',args.pred_dir)
    dummy,_,_,_=load('dummy',args.pred_dir)
    dD=np.load(os.path.join(args.pred_dir,'label_D_test.npz'),allow_pickle=True)
    labD_map={t:v for t,v in zip(dD['frames'],dD['labelD'])}
    eg=np.load(os.path.join(args.cache_dir,'frames_test.npz'),allow_pickle=True)
    ego_map={t:e for t,e in zip(eg['frames'],eg['ego_raw'])}

    labD=np.array([labD_map.get(t,10.0) for t in frames])
    rot =np.array([abs(ego_map[t][ROT_IDX]) if t in ego_map else 0.0 for t in frames])
    straight = rot<=TURN
    riskC = labels<DANGER; riskD = labD<DANGER
    agree = riskC & riskD

    idx = np.where(straight)[0]                 # 직진 프레임
    bl  = agree[idx].astype(int)                # 양성 = C·D 합의 위험
    n_pos = int(bl.sum()); n_sc=len(set(scenes[idx].tolist()))
    print(f"직진 프레임 {len(idx)}, C·D합의 위험 {n_pos}, 장면 {n_sc}")
    if n_pos==0:
        print("합의 위험 프레임 0 → AP 계산 불가"); return

    sc_sub=scenes[idx]; uniq=sorted(set(sc_sub.tolist()))
    s2i={s:idx[np.where(sc_sub==s)[0]] for s in uniq}
    rng=np.random.default_rng(42); deltas=[]
    def mean_ap(ii,preds):
        b=agree[ii].astype(int)
        if b.sum()==0: return np.nan
        return float(np.mean([average_precision_score(b,-preds[k][ii]) for k in range(len(SEEDS))]))
    for _ in range(args.n_boot):
        samp=rng.choice(len(uniq),size=len(uniq),replace=True)
        ii=np.concatenate([s2i[uniq[j]] for j in samp])
        deltas.append(mean_ap(ii,full)-mean_ap(ii,dummy))
    d=np.array(deltas); d=d[~np.isnan(d)]
    lo,hi=np.percentile(d,[2.5,97.5])
    print("\n===== 표 5.3  'C·D 합의 위험, 직진' 행 =====")
    print(f"  위험/장면 : {n_pos}/{n_sc}")
    print(f"  ΔAP       : {d.mean():+.4f}")
    print(f"  95% CI    : [{lo:+.4f}, {hi:+.4f}]")
    print(f"  P(Δ>0)    : {(d>0).mean():.3f}")


if __name__=='__main__':
    main()