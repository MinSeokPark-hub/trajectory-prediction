import pandas as pd
import numpy as np
import logging

from .base_parser import BaseParser

logger = logging.getLogger(__name__)


class SGANParser(BaseParser):
    def __init__(self, fps: float = 2.5):
        super().__init__(fps)

    def parse_label(self, file_path: str) -> pd.DataFrame:
        logger.info(f"SGAN 데이터 로드: {file_path}")
        df = pd.read_csv(
            file_path, sep=r'\s+', header=None,
            names=['frame', 'track_id', 'pos_x', 'pos_y']
        )

        if df.empty:
            return pd.DataFrame()

        df['type']  = 'Pedestrian'
        df['pos_z'] = df['pos_y']
        df['pos_y'] = 0.0

        for col in ['x_pix', 'y_pix', 'w_pix', 'h_pix']:
            df[col] = 0.0

        df['x']     = df['pos_x']
        df['y']     = df['pos_z']
        df['w']     = 0.0
        df['h']     = 0.0
        df['depth'] = df['pos_z']

        return df[[
            'frame', 'track_id', 'type',
            'x', 'y', 'w', 'h', 'depth',
            'x_pix', 'y_pix', 'w_pix', 'h_pix',
            'pos_x', 'pos_y', 'pos_z',
        ]]

    def calculate_gt_ttc(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(['track_id', 'frame'])
        df['prev_x']     = df.groupby('track_id')['pos_x'].shift(1)
        df['prev_z']     = df.groupby('track_id')['pos_z'].shift(1)
        df['prev_frame'] = df.groupby('track_id')['frame'].shift(1)

        dist      = np.sqrt((df['pos_x'] - df['prev_x'])**2 + (df['pos_z'] - df['prev_z'])**2)
        time_diff = self.dt * (df['frame'] - df['prev_frame'])

        df['velocity']   = dist / time_diff
        current_dist     = np.sqrt(df['pos_x']**2 + df['pos_z']**2)
        df['gt_ttc']     = np.where(df['velocity'] > 0, current_dist / df['velocity'], 100.0)
        return df.dropna(subset=['prev_z'])