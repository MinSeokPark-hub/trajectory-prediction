import torch
import torch.nn as nn

class PedestrianTrajectoryModel(nn.Module):
    def __init__(self):
        super(PedestrianTrajectoryModel, self).__init__()
        
        # 1. 3D CNN 블록
        self.conv3d = nn.Conv3d(in_channels=3, out_channels=16, kernel_size=(3, 3, 3), padding=1)
        self.relu = nn.ReLU()
        self.pool3d = nn.MaxPool3d(kernel_size=(1, 2, 2))
        
        # 특징맵의 공간(가로/세로) 크기를 1x1로 깔끔하게 압축해 주는 레이어 (연결을 위해 추가)
        self.adaptive_pool = nn.AdaptiveAvgPool3d((16, 1, 1))
        
        # 2. Attention 블록
        self.attention = nn.MultiheadAttention(embed_dim=16, num_heads=4, batch_first=True)
        
        # 3. 궤적 예측 (출력) 블록
        self.fc = nn.Linear(in_features=16, out_features=2) 

    def forward(self, x):
        # 1. 영상 텐서가 3D CNN을 통과하며 시공간 특징 추출
        x = self.conv3d(x)
        x = self.relu(x)
        x = self.pool3d(x)
        
        # 2. Attention에 넣기 위해 데이터 형태 다듬기
        x = self.adaptive_pool(x) # 공간 차원 압축
        # (배치, 채널, 프레임, 세로, 가로) -> (배치, 프레임, 채널) 형태로 순서 변경
        x = x.view(x.size(0), x.size(2), x.size(1)) 
        
        # 3. Attention 적용 (가려짐 등 중요한 맥락 파악)
        attn_output, _ = self.attention(x, x, x)
        
        # 4. 마지막 프레임의 특징만 뽑아서 X, Y 좌표 2개 계산
        last_feature = attn_output[:, -1, :] 
        output = self.fc(last_feature)
        
        return output # 이제 드디어 빈손(None)이 아니라 예측된 좌표를 들고 반환합니다!

if __name__ == "__main__":
    model = PedestrianTrajectoryModel()
    print("✅ 3D CNN + Attention 모델 뼈대가 성공적으로 생성되었습니다!\n")
    print(model)