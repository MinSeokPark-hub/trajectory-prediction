import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os

# 우리가 만든 모듈들 임포트
from utils.data_helper import load_all_kitti_data
from utils.dataset import KittiSequenceDataset
from predictors.simple_ttc import SimpleTTC

def train():
    # 1. 환경 설정 및 데이터 로드
    LABEL_DIR = "/workspace/minseok_park/data/kitti/labels/"
    MODEL_SAVE_PATH = "/workspace/minseok_park/models/simple_ttc_v1.pth"
    
    if not os.path.exists("/workspace/minseok_park/models"):
        os.makedirs("/workspace/minseok_park/models")

    # 모든 KITTI 라벨 통합 로드
    full_df = load_all_kitti_data(LABEL_DIR)
    dataset = KittiSequenceDataset(full_df, seq_length=3)
    
    # 학습/검증 데이터 분할 (8:2)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # 2. 모델 및 최적화 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleTTC().to(device)
    criterion = nn.L1Loss() # MAE를 직접 최적화하기 위해 L1Loss 사용
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 3. 학습 루프[cite: 1]
    epochs = 20
    print(f"🚀 학습 시작 (Device: {device}, Total Samples: {len(dataset)})")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device).view(-1, 1)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 검증(Validation) 단계[cite: 1]
        model.eval()
        val_mae = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device).view(-1, 1)
                outputs = model(inputs)
                val_mae += torch.abs(outputs - targets).mean().item()

        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss/len(train_loader):.4f} | Val MAE: {val_mae/len(val_loader):.4f}s")

    # 4. 모델 저장[cite: 1]
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"💾 학습 완료! 모델이 저장되었습니다: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()