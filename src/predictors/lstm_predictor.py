import torch
import torch.nn as nn

class LSTMTrajectoryPredictor(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128, output_dim=3, num_layers=2, pred_horizon=5):
        """
        설계 변경: input_dim 2 -> 3 (x, y, depth)
        """
        super(LSTMTrajectoryPredictor, self).__init__()
        self.pred_horizon = pred_horizon
        self.hidden_dim = hidden_dim
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # 출력: pred_horizon(5프레임) * 3차원(x, y, depth)
        self.fc = nn.Linear(hidden_dim, output_dim * pred_horizon)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        last_hidden = hidden[-1]
        
        prediction = self.fc(last_hidden)
        # 여기도 .view 대신 .reshape를 사용합니다.
        return prediction.reshape(-1, self.pred_horizon, 3)

print("✅ 3D LSTM Predictor 모델 클래스가 업데이트되었습니다.")