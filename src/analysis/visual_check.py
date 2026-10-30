# src/analysis/visual_check.py
"""
Phase 6 보강 — 위험 프레임 라벨 타당성 시각 확인 뷰어 (논문 3.6 / 표 3.5).

test split의 위험 프레임(TTC<3s)을 무작위로 N개 골라, 각 프레임을
[전방 카메라 사진 | 위에서 본 박스 다이어그램] 패널 PNG로 저장한다.
사람이 넘겨보며 타당/외삽오탐/판단불가를 verdicts.csv에 기입하면 표 3.5 완성.

BEV 다이어그램: 자차(파랑, 아래 중앙, 위쪽이 전방) + 후보 객체(회색, 위험 객체는
빨강) + 상대속도 화살표(객체−자차, 닫히는 방향) + 각 객체 TTC.

사용: python src/analysis/visual_check.py --n 40
"""
import argparse, os, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.sat_ttc import compute_ttc_sat, EGO_W, EGO_L, EGO_OFFSET
from src.data.nuscenes_ttc_dataset import _is_valid_category, _quat_to_yaw
from nuscenes.utils.geometry_utils import BoxVisibility

DANGER = 3.0
ROT_IDX = 17
TURN_TH = 0.08


def rect_corners(cx, cy, hl, hw, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    ax = np.array([c, s]); ay = np.array([-s, c]); ctr = np.array([cx, cy])
    return np.array([ctr+ax*hl+ay*hw, ctr+ax*hl-ay*hw,
                     ctr-ax*hl-ay*hw, ctr-ax*hl+ay*hw])   # 시계순


def to_screen(pts):
    """ego frame (x=전방, y=좌) → 화면 (가로=-y: 좌가 왼쪽, 세로=x: 전방 위)."""
    return np.stack([-pts[:,1], pts[:,0]], axis=1)


def candidates(nusc, sample):
    cam_token = sample['data']['CAM_FRONT']
    cam = nusc.get('sample_data', cam_token)
    ep = nusc.get('ego_pose', cam['ego_pose_token'])
    ex, ey = ep['translation'][:2]; eyaw = _quat_to_yaw(ep['rotation'])
    pt, nt = sample['prev'], sample['next']
    evx_g = evy_g = 0.0
    if pt and nt:
        def fp(stk):
            s = nusc.get('sample', stk); c = nusc.get('sample_data', s['data']['CAM_FRONT'])
            return nusc.get('ego_pose', c['ego_pose_token'])
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
    out = []
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
        out.append(dict(rx=rx, ry=ry, ryaw=ryaw, w=ann['size'][0], l=ann['size'][1],
                        relvx=rvx-evx, relvy=rvy-evy, ttc=ttc,
                        cat=ann['category_name'].split('.')[0]))
    return out, cam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataroot', default='/workspace/data/nuscenes')
    ap.add_argument('--version', default='v1.0-trainval')
    ap.add_argument('--cache_dir', default='cache')
    ap.add_argument('--out', default='visual_check')
    ap.add_argument('--n', type=int, default=40)
    ap.add_argument('--seed', type=int, default=2024)
    args = ap.parse_args()

    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    d = np.load(os.path.join(args.cache_dir, 'frames_test.npz'), allow_pickle=True)
    frames, labels, ego = d['frames'], d['labels'], d['ego_raw']
    risk = np.where(labels < DANGER)[0]
    rng = np.random.default_rng(args.seed)
    sel = rng.choice(risk, size=min(args.n, len(risk)), replace=False)
    os.makedirs(args.out, exist_ok=True)

    rows = []
    for n, idx in enumerate(sel):
        st = frames[idx]
        sample = nusc.get('sample', st)
        cands, cam = candidates(nusc, sample)
        seg = 'turn' if abs(ego[idx, ROT_IDX]) > TURN_TH else 'straight'
        minttc = float(labels[idx])

        fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.2))
        # 좌: 전방 사진
        try:
            img = Image.open(os.path.join(nusc.dataroot, cam['filename'])).convert('RGB')
            a1.imshow(img)
        except Exception:
            a1.text(0.5, 0.5, 'image missing', ha='center')
        a1.axis('off'); a1.set_title(f'CAM_FRONT  (#{n}  {seg}  minTTC={minttc:.2f}s)')
        # 우: BEV
        ego_c = rect_corners(EGO_OFFSET, 0, EGO_L/2, EGO_W/2, 0.0)
        a2.add_patch(Polygon(to_screen(ego_c), closed=True, fc='#2E6DA4', ec='k', alpha=.8))
        a2.annotate('', xy=to_screen(np.array([[6,0]]))[0], xytext=(0,0),
                    arrowprops=dict(arrowstyle='->', color='#2E6DA4'))
        for c in cands:
            is_risk = c['ttc'] < DANGER
            cc = rect_corners(c['rx'], c['ry'], c['l']/2, c['w']/2, c['ryaw'])
            a2.add_patch(Polygon(to_screen(cc), closed=True,
                                 fc=('#D9534F' if is_risk else '#BBBBBB'),
                                 ec='k', alpha=.75))
            p = to_screen(np.array([[c['rx'], c['ry']]]))[0]
            v = to_screen(np.array([[c['relvx'], c['relvy']]]))[0]  # 방향만
            a2.annotate('', xy=(p[0]+v[0], p[1]+v[1]), xytext=(p[0], p[1]),
                        arrowprops=dict(arrowstyle='->', color='r' if is_risk else 'gray', lw=1.3))
            a2.text(p[0], p[1], f"{c['cat']}\n{c['ttc']:.1f}s", fontsize=7, ha='center',
                    color='darkred' if is_risk else 'dimgray')
        a2.set_xlim(-30, 30); a2.set_ylim(-5, 55); a2.set_aspect('equal')
        a2.axhline(0, color='k', lw=.3); a2.axvline(0, color='k', lw=.3)
        a2.set_title('위에서 본 배치 (위=전방, 빨강=위험객체, 화살표=상대속도)')
        a2.set_xlabel('← 좌   (m)   우 →'); a2.set_ylabel('전방 (m)')
        plt.tight_layout()
        fn = f'panel_{n:03d}_{seg}_ttc{minttc:.1f}.png'
        plt.savefig(os.path.join(args.out, fn), dpi=90); plt.close()
        rows.append((n, st, seg, f'{minttc:.2f}', ''))
        if (n+1) % 10 == 0:
            print(f'  {n+1}/{len(sel)} 패널 생성')

    with open(os.path.join(args.out, 'verdicts.csv'), 'w') as f:
        f.write('idx,frame_token,segment,minTTC,verdict(valid/false/unsure)\n')
        for r in rows:
            f.write(','.join(map(str, r)) + '\n')
    print(f"\n[완료] {len(sel)}개 패널 → {args.out}/  (verdicts.csv에 판정 기입)")
    print("판정 기준: valid=실제 충돌경로, false=직선외삽 오탐, unsure=판단불가")


if __name__ == '__main__':
    main()