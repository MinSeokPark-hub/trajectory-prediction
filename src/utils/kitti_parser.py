import pandas as pd
import numpy as np
import logging

from .base_parser import BaseParser

logger = logging.getLogger(__name__)


class KittiParser(BaseParser):
    def __init__(self, fps: float = 10.0):
        super().__init__(fps)
        self.cols = [
            'frame', 'track_id', 'type', 'truncated', 'occluded', 'alpha',
            'bbox_left', 'bbox_top', 'bbox_right', 'bbox_bottom',
            'height', 'width', 'length', 'pos_x', 'pos_y', 'pos_z', 'rot_y'
        ]
        self.target_types = ['Pedestrian', 'Cyclist', 'Car', 'Van', 'Truck']

    def parse_label(self, file_path: str) -> pd.DataFrame:
        df = pd.read_csv(file_path, sep=' ', header=None, names=self.cols)
        obj_df = df[df['type'].isin(self.target_types)].copy()

        if obj_df.empty:
            logger.warning(f"대상 객체 없음: {file_path}")
            return pd.DataFrame()

        obj_df['x_pix'] = (obj_df['bbox_left'] + obj_df['bbox_right']) / 2
        obj_df['y_pix'] = (obj_df['bbox_top']  + obj_df['bbox_bottom']) / 2
        obj_df['w_pix'] = obj_df['bbox_right']  - obj_df['bbox_left']
        obj_df['h_pix'] = obj_df['bbox_bottom'] - obj_df['bbox_top']

        obj_df['x']     = obj_df['x_pix']
        obj_df['y']     = obj_df['y_pix']
        obj_df['w']     = obj_df['w_pix']
        obj_df['h']     = obj_df['h_pix']
        obj_df['depth'] = obj_df['pos_z']

        return obj_df[[
            'frame', 'track_id', 'type',
            'x', 'y', 'w', 'h', 'depth',
            'x_pix', 'y_pix', 'w_pix', 'h_pix',
            'pos_x', 'pos_y', 'pos_z',
        ]]

    def calculate_gt_ttc(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(['track_id', 'frame'])
        df['prev_z']     = df.groupby('track_id')['pos_z'].shift(1)
        df['prev_frame'] = df.groupby('track_id')['frame'].shift(1)

        time_diff      = self.dt * (df['frame'] - df['prev_frame'])
        df['velocity'] = (df['prev_z'] - df['pos_z']) / time_diff
        df['gt_ttc']   = np.where(df['velocity'] > 0, df['pos_z'] / df['velocity'], 100.0)
        return df.dropna(subset=['prev_z'])