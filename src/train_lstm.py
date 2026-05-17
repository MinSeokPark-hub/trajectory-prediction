import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import joblib
import os

from utils.data_helper import load_all_kitti_data
from utils.lstm_dataset import KittiLSTMDataset
from predictors.lstm_predictor import LSTMTrajectoryPredictor

def train_lstm_3d():
    # 1. 경로 설정
    LABEL_DIR = "/workspace/minseok_park/data/kitti/labels/"
    MODEL_SAVE_PATH = "/workspace/minseok_park/models/lstm_3d_v2.pth"
    SCALER_PATH = "/workspace/minseok_park/models/lstm_scaler_3d.pkl"
    
    if not os.path.exists("/workspace/minseok_park/models"):
        os.makedirs("/workspace/minseok_park/models")

    # 2. 데이터 로드 및 3D 정규화
    full_df = load_all_kitti_data(LABEL_DIR)
    
    # x, y, depth 모두 정규화 대상에 포함
    features = ['x', 'y', 'depth']
    scaler = StandardScaler()
    full_df[features] = scaler.fit_transform(full_df[features])
    joblib.dump(scaler, SCALER_PATH)

    # 3. 데이터셋 생성 (3D 지원 버전)
    dataset = KittiLSTMDataset(full_df, seq_len=5, pred_len=5)
    train_size = int(0.8 * len(dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset)-train_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # 4. 모델 설정 (input_dim=3, output_dim=3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMTrajectoryPredictor(input_dim=3, hidden_dim=128, output_dim=3, num_layers=2).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 5. 학습 실행
    epochs = 30
    print(f"🚀 3D LSTM 학습 시작! (Samples: {len(dataset)}, Device: {device})")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                val_loss += criterion(model(inputs), targets).item()

        print(f"Epoch [{epoch+1}/{30}] | Train: {train_loss/len(train_loader):.6f} | Val: {val_loss/len(val_loader):.6f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"💾 3D LSTM 모델 저장 완료: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_lstm_3d()