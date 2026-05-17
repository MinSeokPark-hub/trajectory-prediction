import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import joblib # 스케일러 저장을 위해 필요

from utils.data_helper import load_all_kitti_data
from utils.dataset import KittiSequenceDataset
from predictors.simple_ttc import SimpleTTC

def train_v2():
    LABEL_DIR = "/workspace/minseok_park/data/kitti/labels/"
    full_df = load_all_kitti_data(LABEL_DIR)
    
    # 1. TTC Clipping: 10초 이상의 위험은 모두 '안전'으로 간주
    full_df['gt_ttc'] = full_df['gt_ttc'].clip(upper=10.0)
    
    # 2. 데이터 정규화 (Input Features)
    features = ['x', 'y', 'w', 'h', 'depth']
    scaler = StandardScaler()
    full_df[features] = scaler.fit_transform(full_df[features])
    
    # 나중에 추론 서버에서 쓰기 위해 스케일러 저장[cite: 1]
    joblib.dump(scaler, "/workspace/minseok_park/models/scaler.pkl")

    dataset = KittiSequenceDataset(full_df, seq_length=3)
    train_size = int(0.8 * len(dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset)-train_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleTTC().to(device)
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"🚀 정규화 적용 학습 시작! (Max TTC: 10s)")
    for epoch in range(30): # 조금 더 길게 학습
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device).view(-1, 1)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_mae = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device).view(-1, 1)
                val_mae += torch.abs(model(inputs) - targets).mean().item()

        print(f"Epoch [{epoch+1}/30] | Loss: {train_loss/len(train_loader):.4f} | Val MAE: {val_mae/len(val_loader):.4f}s")

    torch.save(model.state_dict(), "/workspace/minseok_park/models/simple_ttc_v2_norm.pth")

if __name__ == "__main__":
    train_v2()