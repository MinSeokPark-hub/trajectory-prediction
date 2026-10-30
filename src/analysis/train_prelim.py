# src/analysis/train_prelim.py
"""
Phase 6 보강 — 예비 실험 4.1~4.3 (논문 4장).

--exp 41 : 객체 단위 라벨의 모호성
           객체마다 라벨을 두고 학습 → 같은 프레임의 서로 다른 객체 타깃에
           대해 모델 예측이 거의 같아짐(정답 평균으로 몰림). 예측 분산으로 확인.
--exp 42 : 평균 오차 지표의 한계
           가중 없는 평L1 손실로 학습 → test MAE·위험재현율 보고
           (MAE는 낮지만 재현율도 낮음을 확인).
--exp 43 : 체크포인트 선택 편향
           정상(가중) 학습을 에폭별로 저장하며 에폭별 검증 재현율 기록
           → 검증손실 최저 에폭 vs 마지막 에폭의 재현율 비교.

프레임 캐시(cache/frames_*.npz)와 ego_stats를 재사용. GPU 1개 사용.

사용 예:
  python src/analysis/train_prelim.py --exp 43 --seed 42
"""
import argparse, os, sys, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.nuscenes_ttc_dataset import (
    NuScenesTTCDataset, _is_valid_category, _quat_to_yaw)
from src.models.resnet_ttc import build_model
from src.utils.sat_ttc import compute_ttc_sat, TTC_MAX
from nuscenes.utils.geometry_utils import BoxVisibility

DANGER = 3.0


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


class WeightedL1(nn.Module):
    def forward(self, p, t):
        w = torch.where(t < 3.0, 17.53, torch.where(t < 5.0, 6.96, 0.36))
        return (w * torch.abs(p - t)).mean()


# ── 객체 단위 라벨 데이터셋 (4.1) — 프레임 데이터셋 재사용 ──
class ObjectTTCDataset(Dataset):
    def __init__(self, base: NuScenesTTCDataset):
        self.base = base
        self.pairs = []            # (frame_idx, obj_ttc)
        nusc = base.nusc
        print(f"[Obj:{base.split}] 객체 단위 라벨 계산...")
        for fi, st in enumerate(base.frames):
            sample = nusc.get('sample', st)
            cam_token = sample['data']['CAM_FRONT']
            cam = nusc.get('sample_data', cam_token)
            ep = nusc.get('ego_pose', cam['ego_pose_token'])
            ex, ey = ep['translation'][:2]; eyaw = _quat_to_yaw(ep['rotation'])
            pt, nt = sample['prev'], sample['next']
            def fp(stk):
                s = nusc.get('sample', stk); c = nusc.get('sample_data', s['data']['CAM_FRONT'])
                return nusc.get('ego_pose', c['ego_pose_token'])
            evx_g = evy_g = 0.0
            if pt and nt:
                pp, npz = fp(pt), fp(nt); dt = (npz['timestamp']-pp['timestamp'])*1e-6
                if dt > 1e-3:
                    evx_g = (npz['translation'][0]-pp['translation'][0])/dt
                    evy_g = (npz['translation'][1]-pp['translation'][1])/dt
            try:
                _, boxes, _ = nusc.get_sample_data(cam_token, box_vis_level=BoxVisibility.ANY)
                vis = {b.token for b in boxes}
            except Exception:
                vis = set()
            cy, sy = np.cos(-eyaw), np.sin(-eyaw)
            for at in sample['anns']:
                if at not in vis: continue
                ann = nusc.get('sample_annotation', at)
                if not _is_valid_category(ann['category_name']): continue
                if ann.get('num_lidar_pts', 0) < 1: continue
                try: vel = nusc.box_velocity(at)
                except Exception: continue
                if vel is None or np.any(np.isnan(vel[:2])): continue
                ox, oy = ann['translation'][:2]; dx, dy = ox-ex, oy-ey
                rx = dx*cy-dy*sy; ry = dx*sy+dy*cy
                ryaw = _quat_to_yaw(ann['rotation'])-eyaw
                rvx = vel[0]*cy-vel[1]*sy; rvy = vel[0]*sy+vel[1]*cy
                evx = evx_g*cy-evy_g*sy; evy = evx_g*sy+evy_g*cy
                ttc = compute_ttc_sat(rx, ry, ryaw, ann['size'][0], ann['size'][1],
                                      rvx, rvy, evx, evy, ego_yaw=0.0)
                if ttc > 1e-6:
                    self.pairs.append((fi, float(ttc)))
        print(f"[Obj:{base.split}] 객체 샘플 {len(self.pairs)}")

    def __len__(self): return len(self.pairs)
    def __getitem__(self, i):
        fi, ttc = self.pairs[i]
        img = self.base._get_images(self.base.frames[fi])
        ego = (self.base.ego_raw[fi]-self.base.ego_mean)/self.base.ego_std
        return img, torch.tensor(ego, dtype=torch.float32), torch.tensor(ttc, dtype=torch.float32)


@torch.no_grad()
def collect_preds(model, loader, device):
    model.eval(); P, L = [], []
    for img, ego, lab in loader:
        P.append(model(img.to(device), ego.to(device)).cpu().numpy()); L.append(lab.numpy())
    return np.concatenate(P), np.concatenate(L)


def recall(preds, labels):
    pos = labels < DANGER
    return float(((preds < DANGER) & pos).sum()/max(pos.sum(),1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', required=True, choices=['41','42','43'])
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--dataroot', default='/workspace/data/nuscenes')
    ap.add_argument('--version', default='v1.0-trainval')
    ap.add_argument('--cache_dir', default='cache')
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--workers', type=int, default=6)
    args = ap.parse_args()
    seed_all(args.seed)

    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    s = np.load(os.path.join(args.cache_dir, 'ego_stats.npz'))
    stats = (s['mean'], s['std'])
    tr = NuScenesTTCDataset(nusc, 'train', os.path.join(args.cache_dir,'frames_train.npz'), ego_stats=stats)
    va = NuScenesTTCDataset(nusc, 'val',   os.path.join(args.cache_dir,'frames_val.npz'),   ego_stats=stats)
    te = NuScenesTTCDataset(nusc, 'test',  os.path.join(args.cache_dir,'frames_test.npz'),  ego_stats=stats)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 데이터셋/손실 선택
    if args.exp == '41':
        tr_ds, va_ds = ObjectTTCDataset(tr), ObjectTTCDataset(va)
        crit = WeightedL1()
    else:
        tr_ds, va_ds = tr, va
        crit = (nn.L1Loss() if args.exp == '42' else WeightedL1())

    trL = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers)
    vaL = DataLoader(va_ds, batch_size=32, shuffle=False, num_workers=args.workers)
    teL = DataLoader(te,    batch_size=32, shuffle=False, num_workers=args.workers)

    model = build_model('full').to(device)
    opt = torch.optim.Adam(filter(lambda p:p.requires_grad, model.parameters()), lr=5e-5)

    epoch_stats = []
    for ep in range(1, 4):
        if ep == 1:
            for p in model.image_encoder.parameters(): p.requires_grad = False
            opt = torch.optim.Adam(filter(lambda p:p.requires_grad, model.parameters()), lr=5e-5)
        elif ep == 2:
            for p in model.image_encoder.parameters(): p.requires_grad = True
            opt = torch.optim.Adam(model.parameters(), lr=5e-5)
        model.train(); tl = 0.0
        for img, ego, lab in trL:
            opt.zero_grad(); loss = crit(model(img.to(device), ego.to(device)), lab.to(device))
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); tl += loss.item()
        vp, vlb = collect_preds(model, vaL, device)
        vl = float(np.abs(vp-vlb).mean()); vr = recall(vp, vlb)
        epoch_stats.append((ep, tl/len(trL), vl, vr))
        print(f"  Epoch {ep}  TrainLoss={tl/len(trL):.4f}  ValMAE={vl:.4f}  ValRecall={vr:.4f}")

    print("\n" + "="*52)
    if args.exp == '41':
        vp, vlb = collect_preds(model, vaL, device)
        print("[4.1 객체단위 라벨] 검증 예측 분포")
        print(f"  라벨 std = {vlb.std():.3f},  예측 std = {vp.std():.3f}")
        print(f"  라벨 평균 = {vlb.mean():.3f},  예측 평균 = {vp.mean():.3f}")
        print("  → 예측 std가 라벨 std보다 크게 작으면 '평균으로 몰림'")
    elif args.exp == '42':
        tp, tlb = collect_preds(model, teL, device)
        print("[4.2 무가중 손실] test")
        print(f"  MAE = {np.abs(tp-tlb).mean():.4f},  위험재현율 = {recall(tp,tlb):.4f}")
        print("  → MAE 낮아도 재현율 낮으면 MAE가 부적절 지표라는 근거")
    else:
        best = min(epoch_stats, key=lambda x: x[2])   # 검증 MAE(손실 대용) 최저
        last = epoch_stats[-1]
        print("[4.3 체크포인트 편향]")
        print(f"  검증최저 에폭 {best[0]}: ValRecall={best[3]:.4f}")
        print(f"  마지막 에폭 {last[0]}: ValRecall={last[3]:.4f}")
        print("  → 검증최저 에폭의 재현율이 더 낮으면 best선택이 편향")
    print("="*52)


if __name__ == '__main__':
    main()