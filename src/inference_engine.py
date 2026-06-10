import torch
import joblib
import numpy as np
import pandas as pd
import warnings
from predictors.simple_ttc import SimpleTTC
from predictors.lstm_predictor import LSTMTrajectoryPredictor
from predictors.social_attention import SocialAttentionModule

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
        self.social_attention = SocialAttentionModule()

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

    def predict_scene(self, objects: list[dict], fps: float = 10.0) -> dict:
        """
        US-10 + US-09 + US-13: 프레임 내 모든 객체를 한 번에 처리.

        objects: 리스트, 각 원소는 dict
            {
              'track_id': str,
              'rel_x': float,
              'rel_y': float,
              'vel_x': float,
              'vel_y': float,
              'depth': float,
              'history_3d': np.ndarray (T, 3)  # [pos_x, pos_y, pos_z] 시계열
            }
        fps: 데이터셋 FPS (T+1초 예측 스텝 계산에 사용)

        반환:
            {
              'social': dict          SocialAttentionModule.compute() 결과
              'predictions': list     객체별 예측 결과 (track_id, t1_pos, ttc, status)
              'ade_no_social': float  Social Attention 미적용 ADE (비교용)
              'ade_social': float     Social Attention 적용 ADE (비교용)
            }
        """
        social_result = self.social_attention.compute(objects)
        attention_weights = social_result['attention_weights']

        predictions = []
        t1_step = max(1, round(fps))  # T+1초에 해당하는 예측 스텝 인덱스

        for i, obj in enumerate(objects):
            history = obj.get('history_3d')
            if history is None or len(history) < 2:
                predictions.append({
                    'track_id': obj['track_id'],
                    't1_pos': None,
                    'pred_traj': None,
                    'ttc': 10.0,
                    'status': 'Safe',
                    'attention_weight': float(attention_weights[i]) if len(attention_weights) > i else 0.0,
                })
                continue

            attn_w = float(attention_weights[i]) if len(attention_weights) > i else 1.0

            with torch.no_grad():
                lstm_df = pd.DataFrame(history[-5:], columns=['x', 'y', 'depth'])
                lstm_norm = self.scaler.transform(lstm_df)
                lstm_input = torch.tensor(lstm_norm, dtype=torch.float32).unsqueeze(0).to(self.device)
                pred_norm = self.lstm_model(lstm_input).reshape(5, 3).cpu().numpy()
                pred_real = self.scaler.inverse_transform(pred_norm)

            # T+1초 위치 (데이터셋 FPS 기준)
            t1_idx = min(t1_step - 1, len(pred_real) - 1)
            t1_pos = pred_real[t1_idx].tolist()

            curr_x, curr_z = history[-1, 0], history[-1, 2]
            vx = (pred_real[-1, 0] - curr_x) / (5 * self.dt)
            vz = (pred_real[-1, 2] - curr_z) / (5 * self.dt)
            if vz >= 0:
                ttc = 10.0
            else:
                ttc_z = -curr_z / vz
                pred_x_at_coll = curr_x + vx * ttc_z
                ttc = ttc_z if abs(pred_x_at_coll) < self.safety_margin else 10.0

            ttc = float(np.clip(ttc, 0.1, 10.0))
            status = 'Danger' if ttc <= 1.5 else ('Warning' if ttc <= 3.0 else 'Safe')

            predictions.append({
                'track_id': obj['track_id'],
                't1_pos': t1_pos,
                'pred_traj': pred_real.tolist(),
                'ttc': ttc,
                'status': status,
                'attention_weight': attn_w,
            })

        # ADE/FDE 비교 (Social Attention 적용 전/후) — T14
        ade_no_social, ade_social, fde_no_social, fde_social = self._compute_ade_comparison(objects, predictions, fps)

        return {
            'social': social_result,
            'predictions': predictions,
            'ade_no_social': ade_no_social,
            'ade_social': ade_social,
            'fde_no_social': fde_no_social,
            'fde_social': fde_social,
        }

    def _compute_ade_comparison(self, objects, predictions, fps):
        """
        Social Attention 미적용(균등 가중치) vs 적용 ADE/FDE 근사 비교 (T14)
        실제 미래 위치가 없으므로 예측 분산을 오차 대리 지표로 사용.
        ADE: 전체 스텝 평균 변위 / FDE: 마지막 스텝 변위
        """
        if not predictions:
            return 0.0, 0.0, 0.0, 0.0

        uniform_weight = 1.0 / max(len(objects), 1)
        ade_no_s, ade_s, fde_no_s, fde_s = [], [], [], []

        for pred in predictions:
            if pred['pred_traj'] is None:
                continue
            traj = np.array(pred['pred_traj'])
            displacements = np.linalg.norm(np.diff(traj[:, :2], axis=0), axis=1)
            if len(displacements) == 0:
                continue

            mean_disp  = displacements.mean()
            final_disp = displacements[-1]
            w = pred['attention_weight']
            n = len(objects)

            ade_no_s.append(mean_disp  * uniform_weight * n)
            ade_s.append(mean_disp     * w              * n)
            fde_no_s.append(final_disp * uniform_weight * n)
            fde_s.append(final_disp    * w              * n)

        def _mean(lst): return float(np.mean(lst)) if lst else 0.0
        return _mean(ade_no_s), _mean(ade_s), _mean(fde_no_s), _mean(fde_s)