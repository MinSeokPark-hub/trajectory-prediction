import torch
from torch.utils.data import Dataset
import numpy as np

class KittiLSTMDataset(Dataset):
    def __init__(self, df, seq_len=5, pred_len=5):
        """
        KITTI 데이터를 LSTM 학습용 시퀀스로 변환하는 클래스입니다.
        
        Args:
            df (pd.DataFrame): 전처리 및 정규화가 완료된 KITTI 데이터프레임
            seq_len (int): 과거 관측 프레임 수 (Input)
            pred_len (int): 미래 예측 프레임 수 (Target)
        """
        self.inputs = []
        self.targets = []
        
        # track_id별로 독립적인 시퀀스 생성 (다른 객체끼리 섞이지 않도록 방지)
        for tid in df['track_id'].unique():
            group = df[df['track_id'] == tid]
            
            # x, y, depth 세 가지 정보를 사용하여 3D 궤적을 학습
            # TTC 계산의 핵심인 '거리 변화량'을 학습하기 위해 depth가 필수입니다.
            coords = group[['x', 'y', 'depth']].values
            
            # (과거 5프레임 -> 미래 5프레임) 슬라이딩 윈도우 적용
            # 데이터 총 길이가 (seq_len + pred_len)보다 짧으면 스킵합니다.
            if len(coords) < (seq_len + pred_len):
                continue

            for i in range(len(coords) - seq_len - pred_len + 1):
                # 입력: t-4, t-3, t-2, t-1, t 시점의 (x, y, depth)
                self.inputs.append(coords[i : i + seq_len])
                # 정답: t+1, t+2, t+3, t+4, t+5 시점의 (x, y, depth)
                self.targets.append(coords[i + seq_len : i + seq_len + pred_len])
                
        self.inputs = np.array(self.inputs, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.float32)

        # 데이터가 하나도 없을 경우를 대비한 체크
        if len(self.inputs) == 0:
            print(f"⚠️ 경고: '{tid}' 기반 데이터셋 생성 결과가 0건입니다. 시퀀스 길이를 확인하세요.")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # PyTorch 텐서로 변환하여 반환
        return torch.from_numpy(self.inputs[idx]), torch.from_numpy(self.targets[idx])

if __name__ == "__main__":
    print("✅ 3D(x, y, depth) 지원 LSTM Dataset 클래스 작성이 완료되었습니다.")