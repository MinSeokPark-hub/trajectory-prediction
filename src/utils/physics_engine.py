import numpy as np

class PhysicsEngine:
    def __init__(self, fps=10):
        self.dt = 1.0 / fps

    def calculate_ttc(self, current_depth, predicted_coords):
        """
        predicted_coords: LSTM이 예측한 미래 5프레임의 (x, y, depth)
        """
        # 1. 미래 0.5초 뒤(5번째 프레임)의 예측 depth 추출
        future_depth = predicted_coords[-1, 2] 
        
        # 2. 상대 속도 계산 (현재 거리 - 미래 거리) / 시간 간격
        # 보행자가 다가오면 양수(+) 값이 나옵니다.
        time_horizon = self.dt * len(predicted_coords)
        velocity = (current_depth - future_depth) / time_horizon
        
        # 3. TTC 계산: 현재 거리 / 상대 속도
        if velocity <= 0.01: # 멀어지거나 정지한 경우 안전값 반환
            return 10.0
            
        ttc = current_depth / velocity
        return float(np.clip(ttc, 0, 10.0)) # 0~10초 사이로 제한