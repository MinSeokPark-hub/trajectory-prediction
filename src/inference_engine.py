import torch
import joblib
import numpy as np
import pandas as pd
import warnings
from predictors.simple_ttc import SimpleTTC
from predictors.lstm_predictor import LSTMTrajectoryPredictor

warnings.filterwarnings("ignore", category=UserWarning)

class InferenceEngine:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dt = 0.1
        self.scaler = joblib.load("/workspace/minseok_park/models/lstm_scaler_3d.pkl")
        
        # SimpleTTC 로드 (학습 차원: 15 = 3프레임 x 5피처)
        self.simple_model = SimpleTTC().to(self.device)
        self.simple_model.load_state_dict(torch.load("/workspace/minseok_park/models/simple_ttc_v2_norm.pth"))
        self.simple_model.eval()
        
        # LSTM 로드 (학습 차원: 3D Trajectory)
        self.lstm_model = LSTMTrajectoryPredictor(input_dim=3, output_dim=3).to(self.device)
        self.lstm_model.load_state_dict(torch.load("/workspace/minseok_park/models/lstm_3d_v2.pth"))
        self.lstm_model.eval()

        self.safety_margin = 1.8 

    def predict(self, pix_data, real_3d_data):
        """
        pix_data: [x_pix, y_pix, w_pix, h_pix, pos_z] -> SimpleTTC용 (5개 피처)
        real_3d_data: [pos_x, pos_y, pos_z] -> LSTM용 (3개 피처)
        """
        with torch.no_grad():
            # 1. SimpleTTC: 5개 피처 사용 (최근 3프레임)
            simple_input = torch.tensor(pix_data[-3:], dtype=torch.float32).reshape(1, -1).to(self.device)
            ttc_simple = self.simple_model(simple_input).item()

            # 2. LSTM: 3개 피처 사용 (전체 5프레임)
            # 학습 시 사용한 Scaler로 정규화
            lstm_df = pd.DataFrame(real_3d_data, columns=['x', 'y', 'depth'])
            lstm_norm = self.scaler.transform(lstm_df)
            lstm_input = torch.tensor(lstm_norm, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # 미래 궤적 예측 및 역정규화
            pred_norm = self.lstm_model(lstm_input).reshape(5, 3).cpu().numpy()
            pred_real = self.scaler.inverse_transform(pred_norm)
            
            # 3. 벡터 기반 미터(m) 단위 충돌 계산
            curr_x, curr_z = real_3d_data[-1, 0], real_3d_data[-1, 2]
            vx = (pred_real[-1, 0] - curr_x) / (5 * self.dt)
            vz = (pred_real[-1, 2] - curr_z) / (5 * self.dt)
            
            if vz >= 0: ttc_vector = 10.0
            else:
                ttc_z = -curr_z / vz
                pred_x_at_coll = curr_x + (vx * ttc_z)
                ttc_vector = ttc_z if abs(pred_x_at_coll) < self.safety_margin else 10.0

            # 최종 통합 (두 모델의 결과 중 더 위험한 값 선택)
            final_ttc = min(ttc_simple if ttc_simple > 0 else 10.0, ttc_vector)
            final_ttc = float(np.clip(final_ttc, 0.1, 10.0))
            
            status = "Safe"
            if final_ttc <= 1.5: status = "Danger"
            elif final_ttc <= 3.0: status = "Warning"
            
            return final_ttc, status