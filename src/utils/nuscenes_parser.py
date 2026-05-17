import os
import numpy as np
import pandas as pd
import logging

from nuscenes.nuscenes import NuScenes
from .base_parser import BaseParser

logger = logging.getLogger(__name__)


class NuScenesParser(BaseParser):
    def __init__(self, dataroot: str, version: str = 'v1.0-mini', fps: float = 2.0):
        super().__init__(fps)
        logger.info(f"nuScenes 로드 중: {dataroot}")
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        self.dataroot = dataroot

    def parse_label(self, file_path: str = None) -> pd.DataFrame:
        rows = []

        for scene in self.nusc.scene:
            scene_name   = scene['name']
            sample_token = scene['first_sample_token']
            frame        = 0

            while sample_token:
                sample = self.nusc.get('sample', sample_token)

                # ── ego_pose (자차 GPS 좌표) ───────────────────
                cam_token = sample['data']['CAM_FRONT']
                cam_data  = self.nusc.get('sample_data', cam_token)
                ego_pose  = self.nusc.get('ego_pose', cam_data['ego_pose_token'])

                ego_x    = ego_pose['translation'][0]
                ego_y    = ego_pose['translation'][1]
                ego_z    = ego_pose['translation'][2]
                img_path = os.path.join(self.dataroot, cam_data['filename'])

                # ── 주변 객체 어노테이션 ──────────────────────
                for ann_token in sample['anns']:
                    ann      = self.nusc.get('sample_annotation', ann_token)
                    category = ann['category_name']

                    if not any(k in category for k in
                               ['pedestrian', 'vehicle.car', 'vehicle.truck',
                                'vehicle.bus', 'vehicle.bicycle']):
                        continue

                    # 객체 절대 3D 위치
                    obj_x = ann['translation'][0]
                    obj_y = ann['translation'][1]
                    obj_z = ann['translation'][2]  # 실제 z(높이) 저장

                    # 자차 기준 상대 좌표
                    rel_x = obj_x - ego_x
                    rel_y = obj_y - ego_y
                    depth = np.sqrt(rel_x**2 + rel_y**2)

                    # 실제 3D 바운딩박스 크기 (w, l, h)
                    obj_w = ann['size'][0]  # width
                    obj_l = ann['size'][1]  # length
                    obj_h = ann['size'][2]  # height

                    track_id = ann['instance_token'][:8]
                    obj_type = 'Pedestrian' if 'pedestrian' in category else 'Vehicle'

                    rows.append({
                        'frame':    frame,
                        'track_id': track_id,
                        'type':     obj_type,
                        # 공통 스키마
                        'x':        rel_x,
                        'y':        rel_y,
                        'w':        obj_w,
                        'h':        obj_h,
                        'depth':    depth,
                        'x_pix':    0.0,
                        'y_pix':    0.0,
                        'w_pix':    obj_w,
                        'h_pix':    obj_h,
                        'pos_x':    rel_x,
                        'pos_y':    rel_y,
                        'pos_z':    depth,
                        # nuScenes 전용
                        'obj_l':    obj_l,
                        'obj_z':    obj_z,
                        'abs_x':    obj_x,
                        'abs_y':    obj_y,
                        'ego_x':    ego_x,
                        'ego_y':    ego_y,
                        'ego_z':    ego_z,
                        'img_path': img_path,
                        'scene':    scene_name,
                        # 객체 회전 (quaternion)
                        'rot_w':    ann['rotation'][0],
                        'rot_x':    ann['rotation'][1],
                        'rot_y':    ann['rotation'][2],
                        'rot_z':    ann['rotation'][3],
                    })

                sample_token = sample['next']
                frame += 1

        if not rows:
            logger.warning("nuScenes: 추출된 객체 없음")
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        logger.info(f"nuScenes 파싱 완료: {len(df)}개 객체")
        return df

    def calculate_gt_ttc(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(['track_id', 'frame'])
        df['prev_depth'] = df.groupby('track_id')['depth'].shift(1)
        df['prev_frame'] = df.groupby('track_id')['frame'].shift(1)

        time_diff      = self.dt * (df['frame'] - df['prev_frame'])
        df['velocity'] = (df['prev_depth'] - df['depth']) / time_diff
        df['gt_ttc']   = np.where(
            df['velocity'] > 0,
            df['depth'] / df['velocity'],
            100.0
        )
        return df.dropna(subset=['prev_depth'])

    def load(self, file_path: str = None) -> pd.DataFrame:
        try:
            df = self.parse_label()
            if df.empty:
                return pd.DataFrame()
            df = self.calculate_gt_ttc(df)
            return df
        except Exception as e:
            logger.error(f"nuScenes 파싱 실패: {e}")
            return pd.DataFrame()