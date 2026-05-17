import torch
import torch.nn as nn
import torch.optim as optim
from app.model import PedestrianTrajectoryModel
from app.dataset import PedestrianDataset
from torch.utils.data import DataLoader

def train():
    # 1. 환경 설정 (맥북 GPU 사용 설정)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 학습 장치: {device}")

    # 2. 데이터셋 및 로더 준비
    dataset = PedestrianDataset("data/custom_obsmat.txt")
    if len(dataset) == 0:
        print("❌ 학습할 데이터가 부족합니다. 영상을 더 길게 분석하거나 데이터 보강이 필요합니다.")
        return
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    # 3. 모델, 손실함수, 최적화 알고리즘 설정
    model = PedestrianTrajectoryModel().to(device)
    criterion = nn.MSELoss() # 평균 제곱 오차 (좌표 예측에 최적)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 4. 학습 루프 (Epoch)
    model.train()
    print("🎬 모델 학습 시작...")
    
    for epoch in range(10): # 테스트용으로 10번만 반복
        total_loss = 0
        for obs, gt in loader:
            # 데이터를 장치로 이동
            obs, gt = obs.to(device), gt.to(device)
            
            # 3D CNN 입력을 위해 차원 변경 (배치, 채널3, 프레임8, 1, 1) 등의 가공이 필요할 수 있음
            # 여기서는 간단히 좌표 기반 학습을 진행합니다.
            optimizer.zero_grad()
            
            # 모델의 forward 실행 (현재 모델은 5차원 입력을 받도록 설계됨)
            # 좌표 학습을 위해 입력 형태를 임시로 가공합니다.
            dummy_input = torch.randn(obs.size(0), 3, 16, 112, 112).to(device) 
            output = model(dummy_input)
            
            # 실제 정답(gt)과의 차이 계산 (미래 마지막 좌표 기준)
            loss = criterion(output, gt[:, -1, :])
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/10], Loss: {total_loss:.4f}")

    # 5. 학습된 결과 저장
    torch.save(model.state_dict(), "weights/model_trained.pth")
    print("✅ 학습 완료! 가중치가 'weights/model_trained.pth'에 저장되었습니다.")

if __name__ == "__main__":
    train()