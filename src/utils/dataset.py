import torch
from torch.utils.data import Dataset
import numpy as np

class KittiSequenceDataset(Dataset):
    def __init__(self, processed_df, seq_length=3):
        self.seq_length = seq_length
        self.inputs = []
        self.targets = []
        
        # track_id별로 데이터를 묶어서 시퀀스 생성
        for tid in processed_df['track_id'].unique():
            group = processed_df[processed_df['track_id'] == tid]
            
            # 5가지 입력값: x, y, w, h, depth
            features = group[['x', 'y', 'w', 'h', 'depth']].values
            ttc_labels = group['gt_ttc'].values
            
            # 슬라이딩 윈도우 방식으로 3프레임씩 묶기
            for i in range(len(features) - seq_length + 1):
                self.inputs.append(features[i : i + seq_length])
                # 마지막 프레임의 TTC를 정답으로 사용[cite: 1]
                self.targets.append(ttc_labels[i + seq_length - 1])
                
        self.inputs = np.array(self.inputs, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # PyTorch 모델 입력 형식: (Sequence, Features) -> (3, 5)[cite: 1]
        return torch.from_numpy(self.inputs[idx]), torch.tensor(self.targets[idx])

print("✅ Dataset 클래스가 준비되었습니다.")