import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class CNN3DDataset(Dataset):
    """
    KITTI 이미지 시퀀스 + GT TTC 라벨을 로드하는 Dataset.
    
    - 연속 T프레임을 BBox 영역으로 크롭 후 리사이즈
    - 출력 텐서: (C, T, H, W) = (3, T, 64, 64)
    """

    def __init__(self, df, image_root, T=4, crop_size=(64, 64)):
        """
        Args:
            df: kitti_parser.load() 결과 DataFrame (gt_ttc 컬럼 포함)
            image_root: '/workspace/minseok_park/data/kitti/images'
            T: 입력 프레임 수
            crop_size: (H, W) 리사이즈 크기
        """
        self.image_root = image_root
        self.T = T
        self.crop_size = crop_size
        self.samples = []  # (seq_id, track_id, frame_list, ttc)

        for seq_track, group in df.groupby(['seq_id', 'track_id']):
            seq_id, track_id = seq_track
            group = group.sort_values('frame').reset_index(drop=True)

            if len(group) < T:
                continue

            for i in range(len(group) - T + 1):
                window = group.iloc[i:i + T]
                ttc = float(window.iloc[-1]['gt_ttc'])
                ttc = min(ttc, 10.0)  # 10초 클리핑
                self.samples.append((seq_id, window, ttc))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq_id, window, ttc = self.samples[idx]
        frames = []

        for _, row in window.iterrows():
            img_path = os.path.join(
                self.image_root, seq_id,
                f"{int(row['frame']):06d}.png"
            )
            img = cv2.imread(img_path)
            if img is None:
                # 이미지 로드 실패 시 zero 패딩
                img = np.zeros((*self.crop_size, 3), dtype=np.uint8)
            else:
                # BBox 크롭
                x1 = max(0, int(row['x_pix'] - row['w_pix'] / 2))
                y1 = max(0, int(row['y_pix'] - row['h_pix'] / 2))
                x2 = min(img.shape[1], int(row['x_pix'] + row['w_pix'] / 2))
                y2 = min(img.shape[0], int(row['y_pix'] + row['h_pix'] / 2))

                if x2 > x1 and y2 > y1:
                    img = img[y1:y2, x1:x2]
                
                img = cv2.resize(img, self.crop_size)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 정규화 [0, 1]
            img = img.astype(np.float32) / 255.0
            frames.append(img)

        # (T, H, W, C) → (C, T, H, W)
        tensor = torch.from_numpy(np.stack(frames, axis=0))  # (T, H, W, C)
        tensor = tensor.permute(3, 0, 1, 2)                  # (C, T, H, W)

        return tensor, torch.tensor(ttc, dtype=torch.float32)