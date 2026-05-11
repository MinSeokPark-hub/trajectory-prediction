import pandas as pd
import numpy as np

class KittiParser:
    def __init__(self, fps=10):
        self.fps = fps
        self.dt = 1.0 / fps
        self.cols = [
            'frame', 'track_id', 'type', 'truncated', 'occluded', 'alpha', 
            'bbox_left', 'bbox_top', 'bbox_right', 'bbox_bottom', 
            'height', 'width', 'length', 'pos_x', 'pos_y', 'pos_z', 'rot_y'
        ]

    def parse_label(self, file_path):
        """보행자, 자전거, 차량(Car, Van, Truck) 데이터를 모두 추출합니다."""
        df = pd.read_csv(file_path, sep=' ', names=self.cols)
        target_types = ['Pedestrian', 'Cyclist', 'Car', 'Van', 'Truck']
        obj_df = df[df['type'].isin(target_types)].copy()
        
        # 시각화 및 SimpleTTC용 픽셀 좌표
        obj_df['x_pix'] = (obj_df['bbox_left'] + obj_df['bbox_right']) / 2
        obj_df['y_pix'] = (obj_df['bbox_top'] + obj_df['bbox_bottom']) / 2
        obj_df['w_pix'] = obj_df['bbox_right'] - obj_df['bbox_left']
        obj_df['h_pix'] = obj_df['bbox_bottom'] - obj_df['bbox_top']
        
        # 모든 필요한 정보를 포함하여 반환
        return obj_df[['frame', 'track_id', 'type', 'x_pix', 'y_pix', 'w_pix', 'h_pix', 'pos_x', 'pos_y', 'pos_z']]

    def calculate_gt_ttc(self, df):
        df = df.sort_values(['track_id', 'frame'])
        df['prev_z'] = df.groupby('track_id')['pos_z'].shift(1)
        df['prev_frame'] = df.groupby('track_id')['frame'].shift(1)
        df['velocity'] = (df['prev_z'] - df['pos_z']) / (self.dt * (df['frame'] - df['prev_frame']))
        df['gt_ttc'] = np.where(df['velocity'] > 0, df['pos_z'] / df['velocity'], 100.0)
        return df.dropna(subset=['prev_z'])