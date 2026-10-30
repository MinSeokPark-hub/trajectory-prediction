# src/analysis/box_sensitivity.py
"""
Phase 6b — 자차 박스 설정별 위험 프레임 수 (논문 표 3.2 / 3.2.4).

후보 객체 기하(자차 로컬 프레임의 위치·속도·크기)를 한 번만 추출해
캐시하고, 자차 박스 크기/오프셋 5개 설정에 대해 SAT TTC를 다시 계산해
3초 위험 프레임 수(전체 / 평가)를 비교한다.

설정 (W×L, offset):
  A    1.85 ×4.5    0.0
  B    1.730×4.084  0.0
  C    1.730×4.084  1.4   (주 분석)
  C13  1.730×4.084  1.3
  C15  1.730×4.084  1.5

사용: python src/analysis/box_sensitivity.py
"""
import argparse
import os
import pickle
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import src.utils.sat_ttc as satmod
from src.data.nuscenes_ttc_dataset import (
    get_scene_splits, _is_valid_category, _quat_to_yaw, N_FRAMES)
from nuscenes.utils.geometry_utils import BoxVisibility

DANGER = 3.0
SETUPS = [
    ('A',   1.85,  4.5,   0.0),
    ('B',   1.730, 4.084, 0.0),
    ('C',   1.730, 4.084, 1.4),
    ('C13', 1.730, 4.084, 1.3),
    ('C15', 1.730, 4.084, 1.5),
]


def ego_velocity_central(nusc, sample):
    pt, nt = sample['prev'], sample['next']
    if not pt or not nt:
        return 0.0, 0.0
    def fp(stok):
        s = nusc.get('sample', stok)
        cam = nusc.get('sample_data', s['data']['CAM_FRONT'])
        return nusc.get('ego_pose', cam['ego_pose_token'])
    pp, npz = fp(pt), fp(nt)
    dt = (npz['timestamp'] - pp['timestamp']) * 1e-6
    if dt <= 1e-3:
        return 0.0, 0.0
    return ((npz['translation'][0] - pp['translation'][0]) / dt,
            (npz['translation'][1] - pp['translation'][1]) / dt)


def extract_geometry(nusc, cache):
    """프레임별 후보 기하를 추출(자차 로컬). 반환: list[(split, [cand,...])]."""
    if os.path.exists(cache):
        with open(cache, 'rb') as f:
            return pickle.load(f)

    splits = get_scene_splits(nusc)
    frame_split = {}
    for sp in ('train', 'val', 'test'):
        for sc in splits[sp]:
            tok = nusc.get('scene', sc)['first_sample_token']
            idxs = []
            while tok:
                idxs.append(tok); tok = nusc.get('sample', tok)['next']
            for i, st in enumerate(idxs):
                if i < N_FRAMES - 1 or i == len(idxs) - 1:
                    continue
                frame_split[st] = sp

    out = []
    print(f"[BoxSens] 후보 기하 추출 ({len(frame_split)}프레임)...")
    for k, (st, sp) in enumerate(frame_split.items()):
        sample = nusc.get('sample', st)
        cam_token = sample['data']['CAM_FRONT']
        cam = nusc.get('sample_data', cam_token)
        epose = nusc.get('ego_pose', cam['ego_pose_token'])
        ex, ey = epose['translation'][:2]
        eyaw = _quat_to_yaw(epose['rotation'])
        evx_g, evy_g = ego_velocity_central(nusc, sample)
        try:
            _, boxes, _ = nusc.get_sample_data(cam_token,
                                               box_vis_level=BoxVisibility.ANY)
            vis = {b.token for b in boxes}
        except Exception:
            vis = set()

        cands = []
        cy, sy = np.cos(-eyaw), np.sin(-eyaw)
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
            dx, dy = ox - ex, oy - ey
            rx = dx * cy - dy * sy
            ry = dx * sy + dy * cy
            ryaw = _quat_to_yaw(ann['rotation']) - eyaw
            rvx = vel[0] * cy - vel[1] * sy
            rvy = vel[0] * sy + vel[1] * cy
            evx = evx_g * cy - evy_g * sy
            evy = evx_g * sy + evy_g * cy
            cands.append((rx, ry, ryaw, ann['size'][0], ann['size'][1],
                          rvx, rvy, evx, evy))
        out.append((sp, cands))
        if (k + 1) % 2000 == 0:
            print(f"  {k+1}/{len(frame_split)}")

    with open(cache, 'wb') as f:
        pickle.dump(out, f)
    print(f"[BoxSens] 캐시 저장: {cache}")
    return out


def min_ttc(cands):
    m = satmod.TTC_MAX
    for c in cands:
        t = satmod.compute_ttc_sat(*c, ego_yaw=0.0)
        if t <= 1e-6:            # 접촉 프레임
            return None
        m = min(m, t)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataroot', default='/workspace/data/nuscenes')
    ap.add_argument('--version', default='v1.0-trainval')
    ap.add_argument('--cache_dir', default='cache')
    args = ap.parse_args()

    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    geo = extract_geometry(nusc, os.path.join(args.cache_dir, 'box_geometry.pkl'))

    print("\n===== 표 3.2  자차 박스 설정별 3초 위험 프레임 (전체 / 평가) =====")
    print(f"{'설정':5} {'박스(W×L)':>14} {'offset':>7} {'위험(전체/평가)':>16}")
    for name, w, l, off in SETUPS:
        satmod.EGO_W, satmod.EGO_L, satmod.EGO_OFFSET = w, l, off
        tot = test = 0
        for sp, cands in geo:
            t = min_ttc(cands)
            if t is None:
                continue
            if t < DANGER:
                tot += 1
                if sp == 'test':
                    test += 1
        print(f"{name:5} {f'{w}×{l}':>14} {off:>7.1f} {f'{tot} / {test}':>16}")


if __name__ == '__main__':
    main()