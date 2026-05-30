import numpy as np


class SocialAttentionModule:
    """
    US-10: 다중 객체 간 Social Self-Attention 모델링
    US-09: 자차 기준 위험도 어텐션 가중치 산출

    학습 없이 물리 기반으로 동작 — 즉시 추론 가능.
    """

    def __init__(self, sigma_dist: float = 10.0, approach_weight: float = 2.0):
        self.sigma_dist = sigma_dist        # 거리 감쇠 파라미터 (m)
        self.approach_weight = approach_weight  # 접근 속도 위험 가중치

    # ── 공개 API ──────────────────────────────────────────────────────────────

    def compute(self, objects: list[dict]) -> dict:
        """
        objects: 리스트, 각 원소는 dict
            {
              'track_id': str,
              'rel_x': float,   # 자차 기준 상대 x (m)
              'rel_y': float,   # 자차 기준 상대 y (m, 전방 양수)
              'vel_x': float,   # x 속도 (m/s)
              'vel_y': float,   # y 속도 (m/s, 접근이면 음수)
              'depth': float,   # 자차까지 거리 (m)
            }

        반환:
            {
              'attention_weights': np.ndarray (N,)   위험도 기준 소프트맥스 가중치
              'danger_scores':     np.ndarray (N,)   원시 위험 점수 (시각화용)
              'interaction_matrix': np.ndarray (N,N) 객체 간 상호작용 행렬
              'evasion_flags':     np.ndarray (N,)   bool, 회피 기동 감지
              'risk_levels':       list[str]         'high'/'mid'/'low'
              'track_ids':         list[str]
            }
        """
        n = len(objects)
        if n == 0:
            return self._empty_result()

        track_ids = [o['track_id'] for o in objects]
        positions = np.array([[o['rel_x'], o['rel_y']] for o in objects], dtype=float)
        velocities = np.array([[o['vel_x'], o['vel_y']] for o in objects], dtype=float)
        depths = np.array([o['depth'] for o in objects], dtype=float)

        danger_scores = self._ego_danger_scores(positions, velocities, depths)
        attention_weights = self._softmax(danger_scores)
        interaction_matrix = self._interaction_matrix(positions, velocities)
        evasion_flags = self._detect_evasion(positions, velocities, interaction_matrix)
        risk_levels = self._classify_risk(attention_weights)

        return {
            'attention_weights': attention_weights,
            'danger_scores': danger_scores,
            'interaction_matrix': interaction_matrix,
            'evasion_flags': evasion_flags,
            'risk_levels': risk_levels,
            'track_ids': track_ids,
        }

    # ── 내부 계산 ──────────────────────────────────────────────────────────────

    def _ego_danger_scores(self, positions, velocities, depths):
        """자차 기준 위험도 점수 — 거리 역수 + 접근 속도 반영 (T4)"""
        dist = np.maximum(depths, 0.5)
        dist_score = 1.0 / dist

        # y 속도가 음수이면 자차 방향으로 접근 중
        approach_vel = np.maximum(-velocities[:, 1], 0.0)
        approach_score = self.approach_weight * approach_vel / (dist + 1e-6)

        return dist_score + approach_score

    def _softmax(self, x):
        e = np.exp(x - x.max())
        return e / (e.sum() + 1e-9)

    def _interaction_matrix(self, positions, velocities):
        """
        N×N 상호작용 행렬 — 거리 + 상대속도 기반 (T7)
        선형 복잡도 유지를 위해 O(N²) 하한 내에서 처리 (T9)
        """
        n = len(positions)
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                diff = positions[j] - positions[i]
                dist = np.linalg.norm(diff) + 1e-6
                rel_vel = velocities[j] - velocities[i]
                # 상대방이 i 방향으로 움직이는 정도
                approach = -np.dot(rel_vel, diff) / (dist ** 2 + 1e-6)
                raw = np.exp(-dist / self.sigma_dist) + 0.5 * max(approach, 0)
                matrix[i, j] = raw

        # 행별 정규화
        row_sum = matrix.sum(axis=1, keepdims=True)
        row_sum = np.where(row_sum < 1e-9, 1.0, row_sum)
        return matrix / row_sum

    def _detect_evasion(self, positions, velocities, interaction_matrix, threshold=0.35):
        """
        회피 기동 감지 — 상호작용 가중치가 높은 인접 객체가 있을 때
        속도 방향 변화(꺾임)를 회피 기동으로 판단 (T8)
        """
        n = len(positions)
        flags = np.zeros(n, dtype=bool)
        for i in range(n):
            # i에게 가장 영향력 있는 객체
            max_influence = interaction_matrix[:, i].max() if n > 1 else 0.0
            if max_influence < threshold:
                continue
            speed = np.linalg.norm(velocities[i])
            if speed < 0.3:
                continue
            # 횡방향(x) 속도 비율이 20% 이상이면 꺾임으로 판단
            lateral_ratio = abs(velocities[i, 0]) / (speed + 1e-6)
            if lateral_ratio > 0.2:
                flags[i] = True
        return flags

    def _classify_risk(self, attention_weights):
        """
        위험 등급 분류 — 상위 1개 객체 가중치가 나머지보다 2배 이상이면 'high' (T5)
        """
        n = len(attention_weights)
        if n == 0:
            return []
        sorted_w = np.sort(attention_weights)[::-1]
        levels = []
        for i, w in enumerate(attention_weights):
            if n >= 2 and w == sorted_w[0] and sorted_w[0] >= 2 * sorted_w[1]:
                levels.append('high')
            elif w >= sorted_w[0] * 0.5:
                levels.append('mid')
            else:
                levels.append('low')
        return levels

    def _empty_result(self):
        return {
            'attention_weights': np.array([]),
            'danger_scores': np.array([]),
            'interaction_matrix': np.zeros((0, 0)),
            'evasion_flags': np.array([], dtype=bool),
            'risk_levels': [],
            'track_ids': [],
        }
