import pandas as pd
import numpy as np

class SGANParser:
    def __init__(self, fps=2.5): 
        # ETH/UCY 데이터셋은 보통 초당 2.5프레임(0.4초 간격)으로 어노테이션 되어 있습니다.
        self.fps = fps
        self.dt = 1.0 / fps

    def parse_label(self, file_path):
        """
        SGAN(ETH/UCY) 데이터 형식: [frame_id, ped_id, pos_x, pos_y]
        """
        print(f"📄 SGAN 데이터 로드 중: {file_path}")
        # 데이터가 탭(\t)이나 공백으로 구분되어 있으므로 정규식 '\s+' 사용
        df = pd.read_csv(file_path, sep='\s+', header=None, 
                         names=['frame', 'track_id', 'pos_x', 'pos_y'])
        
        df['type'] = 'Pedestrian'  # ETH/UCY는 기본적으로 모두 보행자입니다.
        
        # KITTI 파서와의 호환성을 위한 좌표 변환
        # 탑다운 뷰의 y좌표(세로)를 KITTI의 깊이(z좌표)로 매핑합니다.
        df['pos_z'] = df['pos_y'] 
        df['pos_y'] = 0.0         # 지면 높이는 0으로 고정
        
        # ETH/UCY는 이미지 바운딩 박스가 없으므로 픽셀 좌표는 더미(Dummy) 값 처리
        df['x_pix'], df['y_pix'], df['w_pix'], df['h_pix'] = 0, 0, 0, 0
        
        return df

    def calculate_gt_ttc(self, df):
        """탑다운 2D 평면에서의 이동 속도 및 TTC 계산"""
        df = df.sort_values(['track_id', 'frame'])
        df['prev_x'] = df.groupby('track_id')['pos_x'].shift(1)
        df['prev_z'] = df.groupby('track_id')['pos_z'].shift(1)
        df['prev_frame'] = df.groupby('track_id')['frame'].shift(1)
        
        # 2D 평면에서의 유클리디안 이동 거리 (미터)
        dist = np.sqrt((df['pos_x'] - df['prev_x'])**2 + (df['pos_z'] - df['prev_z'])**2)
        time_diff = self.dt * (df['frame'] - df['prev_frame'])
        
        df['velocity'] = dist / time_diff
        
        # 단순화된 TTC (원점 기준 - 필요에 따라 충돌 지점 기준으로 수정 가능)
        current_dist = np.sqrt(df['pos_x']**2 + df['pos_z']**2)
        df['gt_ttc'] = np.where(df['velocity'] > 0, current_dist / df['velocity'], 100.0)
        
        return df.dropna(subset=['prev_z'])