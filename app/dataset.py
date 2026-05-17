import torch
from torch.utils.data import Dataset
import numpy as np
import os

class PedestrianDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        # 1. 텍스트 파일이 있는지 확인
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {data_path}")
            
        # 2. 데이터 로드
        raw_data = np.loadtxt(data_path)
        
        # 3. 학습 설정 (데이터가 적어도 에러 안 나게 짧게 설정)
        self.obs_len = 5  # 과거 5프레임 관찰
        self.pred_len = 5 # 미래 5프레임 예측
        
        # 좌표(X, Y) 컬럼만 추출
        coords = raw_data[:, 2:4] 
        self.data = torch.tensor(coords, dtype=torch.float32)

    def __len__(self):
        # 전체 데이터 개수를 (관찰+예측) 묶음으로 나눈 수
        return max(0, len(self.data) // (self.obs_len + self.pred_len))

    def __getitem__(self, idx):
        start = idx * (self.obs_len + self.pred_len)
        end = start + self.obs_len + self.pred_len
        sequence = self.data[start:end]
        
        observed = sequence[:self.obs_len]
        ground_truth = sequence[self.obs_len:]
        
        return observed, ground_truth