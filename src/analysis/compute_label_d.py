# src/analysis/compute_label_d.py
"""
Phase 5 보강 — CTRV 대안 라벨 D (논문 3.2.4).

라벨 C: 자차가 직선 등속으로 움직인다고 가정 (SAT 닫힌 공식, sat_ttc.py).
라벨 D: 자차가 '일정 회전율·일정 속도(CTRV)'로 원호를 그리며 움직인다고 가정.
        → 시간을 dt로 쪼개 자차를 원호 따라 전진시키며 겹침을 판정(forward sim).

출력:
  - C / D 위험 프레임 수(전체), 합집합, 둘 다 위험(합의) 수와 비율
  - test split의 라벨 D를 preds/label_D_test.npz로 저장(표 5.3 'C·D 합의' 행용)

사용: python src/analysis/compute_label_d.py
"""
import argparse
import os
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.sat_ttc import compute_ttc_sat, EGO_W, EGO_L, EGO_OFFSET, TTC_MAX
from src.data.nuscenes_ttc_dataset import (
    get_scene_splits, _is_valid_category, _quat_to_yaw, N_FRAMES)
from nuscenes.utils.geometry_utils import BoxVisibility

DANGER = 3.0
DT = 0.05            # 시뮬레이션 시간 간격(s)
N_STEP = int(TTC_MAX / DT)


# ── 정적 SAT 겹침 판정 ────────────────────────────────────────
def _rect_corners(cx, cy, hl, hw, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    ax = np.array([c, s]); ay = np.array([-s, c])
    ctr = np.array([cx, cy])
    return np.array([ctr + ax*hl + ay*hw, ctr + ax*hl - ay*hw,
                     ctr - ax*hl + ay*hw, ctr - ax*hl - ay*hw])


def _overlap(c1, c2, yaw1, yaw2):
    for yaw in (yaw1, yaw2):
        cc, ss = np.cos(yaw), np.sin(yaw)
        for axis in (np.array([cc, ss]), np.array([-ss, cc])):
            p1 = c1 @ axis; p2 = c2 @ axis
            if p1.max() < p2.min() or p2.max() < p1.min():
                return False
    return True


# ── 자차 yaw rate 중심차분 ────────────────────────────────────
def ego_yawrate_central(nusc, sample):
    pt, nt = sample['prev'], sample['next']
    if not pt or not nt:
        return 0.0
    def fp(stok):
        s = nusc.get('sample', stok)
        cam = nusc.get('sample_data', s['data']['CAM_FRONT'])
        return nusc.get('ego_pose', cam['ego_pose_token'])
    pp, npz = fp(pt), fp(nt)
    dt = (npz['timestamp'] - pp['timestamp']) * 1e-6
    if dt <= 1e-3:
        return 0.0
    dyaw = _quat_to_yaw(npz['rotation']) - _quat_to_yaw(pp['rotation'])
    dyaw = (dyaw + np.pi) % (2*np.pi) - np.pi      # wrap
    return dyaw / dt


def ttc_ctrv(rx, ry, ryaw, ow, ol, ovx, ovy, v, omega):
    """CTRV 자차 + 직선 객체 forward sim. 반환 TTC(접촉 0, 없으면 TTC_MAX)."""
    hl_e, hw_e = EGO_L/2, EGO_W/2
    hl_o, hw_o = ol/2, ow/2
    for k in range(N_STEP):
        t = k * DT
        th = omega * t
        if abs(omega) > 1e-6:
            ex = (v/omega) * np.sin(th)
            ey = (v/omega) * (1 - np.cos(th))
        else:
            ex, ey = v * t, 0.0
        ecx = ex + EGO_OFFSET * np.cos(th)
        ecy = ey + EGO_OFFSET * np.sin(th)
        ocx, ocy = rx + ovx * t, ry + ovy * t
        ce = _rect_corners(ecx, ecy, hl_e, hw_e, th)
        co = _rect_corners(ocx, ocy, hl_o, hw_o, ryaw)
        if _overlap(ce, co, th, ryaw):
            return t
    return TTC_MAX


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataroot', default='/workspace/data/nuscenes')
    ap.add_argument('--version', default='v1.0-trainval')
    ap.add_argument('--pred_dir', default='preds')
    args = ap.parse_args()

    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    splits = get_scene_splits(nusc)

    nC = nD = both = union = 0
    test_tokens, test_D = [], []

    for sp in ('train', 'val', 'test'):
        for sc in splits[sp]:
            tok = nusc.get('scene', sc)['first_sample_token']
            order = []
            while tok:
                order.append(tok); tok = nusc.get('sample', tok)['next']
            for i, st in enumerate(order):
                if i < N_FRAMES - 1 or i == len(order) - 1:
                    continue
                sample = nusc.get('sample', st)
                cam_token = sample['data']['CAM_FRONT']
                cam = nusc.get('sample_data', cam_token)
                ep = nusc.get('ego_pose', cam['ego_pose_token'])
                ex, ey = ep['translation'][:2]
                eyaw = _quat_to_yaw(ep['rotation'])
                # 자차 속도·yaw rate
                pt, nt = sample['prev'], sample['next']
                def fp(stok):
                    s = nusc.get('sample', stok)
                    cc = nusc.get('sample_data', s['data']['CAM_FRONT'])
                    return nusc.get('ego_pose', cc['ego_pose_token'])
                pp, np_ = fp(pt), fp(nt)
                dt = (np_['timestamp'] - pp['timestamp']) * 1e-6
                evx_g = (np_['translation'][0]-pp['translation'][0])/dt if dt>1e-3 else 0.0
                evy_g = (np_['translation'][1]-pp['translation'][1])/dt if dt>1e-3 else 0.0
                omega = ego_yawrate_central(nusc, sample)
                try:
                    _, boxes, _ = nusc.get_sample_data(cam_token,
                                        box_vis_level=BoxVisibility.ANY)
                    vis = {b.token for b in boxes}
                except Exception:
                    vis = set()
                cy, syw = np.cos(-eyaw), np.sin(-eyaw)
                v_fwd = evx_g*cy - evy_g*syw        # 전방 성분(ego frame x)

                minC, minD = TTC_MAX, TTC_MAX
                contact = False
                for at in sample['anns']:
                    if at not in vis:
                        continue
                    ann = nusc.get('sample_annotation', at)
                    if not _is_valid_category(ann['category_name']):
                        continue
                    if ann.get('num_lidar_pts', 0) < 1:
                        continue
                    try:
                        vel = nusc.box_velocity(at)
                    except Exception:
                        continue
                    if vel is None or np.any(np.isnan(vel[:2])):
                        continue
                    ox, oy = ann['translation'][:2]
                    dx, dy = ox-ex, oy-ey
                    rx = dx*cy - dy*syw; ry = dx*syw + dy*cy
                    ryaw = _quat_to_yaw(ann['rotation']) - eyaw
                    rvx = vel[0]*cy - vel[1]*syw; rvy = vel[0]*syw + vel[1]*cy
                    evx = evx_g*cy - evy_g*syw; evy = evx_g*syw + evy_g*cy
                    # C: 직선 SAT
                    tc = compute_ttc_sat(rx, ry, ryaw, ann['size'][0], ann['size'][1],
                                         rvx, rvy, evx, evy, ego_yaw=0.0)
                    # D: CTRV
                    td = ttc_ctrv(rx, ry, ryaw, ann['size'][0], ann['size'][1],
                                  rvx, rvy, v_fwd, omega)
                    if tc <= 1e-6 or td <= 1e-6:
                        contact = True; break
                    minC = min(minC, tc); minD = min(minD, td)
                if contact:
                    continue
                rC, rD = minC < DANGER, minD < DANGER
                nC += rC; nD += rD
                both += (rC and rD); union += (rC or rD)
                if sp == 'test':
                    test_tokens.append(st); test_D.append(minD)

    print("\n===== CTRV 라벨 D vs 직선 라벨 C (전체, 3초) =====")
    print(f"  C 위험 프레임 : {nC}")
    print(f"  D 위험 프레임 : {nD}")
    print(f"  합집합        : {union}")
    print(f"  둘 다 위험    : {both}  ({100*both/max(union,1):.0f}% 합의)")

    os.makedirs(args.pred_dir, exist_ok=True)
    out = os.path.join(args.pred_dir, 'label_D_test.npz')
    np.savez(out, frames=np.array(test_tokens, dtype=object),
             labelD=np.array(test_D, dtype=np.float32))
    print(f"\n[Saved] {out}")


if __name__ == '__main__':
    main()