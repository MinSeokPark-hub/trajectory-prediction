import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.kitti_parser import KittiParser
from utils.cnn3d_dataset import CNN3DDataset
from predictors.cnn3d_predictor import CNN3DPredictor

def train_cnn3d():
    IMAGE_ROOT  = "/workspace/minseok_park/data/kitti/images"
    LABEL_DIR   = "/workspace/minseok_park/data/kitti/labels"
    MODEL_PATH  = "/workspace/minseok_park/models/cnn3d_ttc.pth"
    T, H, W     = 4, 64, 64
    BATCH_SIZE  = 16
    EPOCHS      = 20
    LR          = 1e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── 데이터 로드 ───────────────────────────────────────────
    parser = KittiParser(fps=10)
    all_dfs = []
    for fname in sorted(os.listdir(LABEL_DIR)):
        if not fname.endswith('.txt'):
            continue
        seq_id = fname.replace('.txt', '')
        df = parser.load(os.path.join(LABEL_DIR, fname))
        if df.empty:
            continue
        df['seq_id'] = seq_id
        all_dfs.append(df)

    import pandas as pd
    full_df = pd.concat(all_dfs, ignore_index=True)
    print(f"총 데이터: {len(full_df)}개")

    # ── Dataset / DataLoader ──────────────────────────────────
    dataset    = CNN3DDataset(full_df, IMAGE_ROOT, T=T, crop_size=(H, W))
    train_size = int(0.8 * len(dataset))
    val_size   = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    print(f"Train: {len(train_ds)} / Val: {len(val_ds)}")

    # ── 모델 ─────────────────────────────────────────────────
    model     = CNN3DPredictor(in_channels=3, T=T, H=H, W=W).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # ── 학습 루프 ─────────────────────────────────────────────
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device).view(-1, 1)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_mae = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device).view(-1, 1)
                val_mae += torch.abs(model(x) - y).mean().item()

        scheduler.step()
        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"Train Loss: {train_loss/len(train_loader):.4f} | "
              f"Val MAE: {val_mae/len(val_loader):.4f}s")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"모델 저장 완료: {MODEL_PATH}")

if __name__ == "__main__":
    train_cnn3d()