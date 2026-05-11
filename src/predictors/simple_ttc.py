import torch
import torch.nn as nn

class SimpleTTC(nn.Module):
    def __init__(self, input_size=15, hidden_size=60, num_layers=8):
        """
        설계서 기준: 입력 5개 값 * 3프레임 = 15
        은닉층 8개, 뉴런 60개
        """
        super(SimpleTTC, self).__init__()
        
        layers = []
        # 입력층 -> 첫 번째 은닉층
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        
        # 나머지 7개의 은닉층 추가 (총 8개)[cite: 1]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            
        # 출력층 (TTC 1개 값 예측)[cite: 1]
        self.feature_extractor = nn.Sequential(*layers)
        self.regressor = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (Batch, 3, 5)
        # .view 대신 .reshape를 사용하여 메모리 연속성 문제를 해결합니다.
        x = x.reshape(x.size(0), -1) 
        x = self.feature_extractor(x)
        return self.regressor(x)

print("✅ SimpleTTC 모델 클래스가 정의되었습니다.[cite: 1]")