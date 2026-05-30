import numpy as np


class OcclusionHandler:
    """
    US-11: 가림 현상(Occlusion) 대응 컨텍스트 강화

    객체가 사라졌을 때 마지막 관성(속도·방향)으로 위치를 유지하고,
    재등장 시 예측 위치와의 오차를 측정한다.
    """

    def __init__(self, max_missing_frames: int = 10, fps: float = 10.0):
        self.max_missing_frames = max_missing_frames
        self.fps = fps
        self.dt = 1.0 / fps

        # track_id → state dict
        self._states: dict[str, dict] = {}

    # ── 공개 API ──────────────────────────────────────────────────────────────

    def update(self, visible_ids: list[str], current_objects: list[dict]) -> list[dict]:
        """
        매 프레임 호출. 현재 보이는 객체들로 상태 갱신 후,
        occlusion 중인 객체의 관성 예측 위치를 함께 반환.

        current_objects: [{'track_id': str, 'rel_x', 'rel_y', 'vel_x', 'vel_y', 'depth'}, ...]

        반환: current_objects + occlusion 중인 객체(가상) 리스트
        """
        # 현재 보이는 객체 상태 갱신
        seen_ids = set()
        for obj in current_objects:
            tid = obj['track_id']
            seen_ids.add(tid)
            self._states[tid] = {
                'rel_x': obj['rel_x'],
                'rel_y': obj['rel_y'],
                'vel_x': obj['vel_x'],
                'vel_y': obj['vel_y'],
                'depth': obj['depth'],
                'missing_frames': 0,
                'occluded': False,
                'predicted_positions': [],
            }

        # 사라진 객체 처리 (T10: 관성 특징 유지)
        virtual_objects = []
        to_delete = []
        for tid, state in self._states.items():
            if tid in seen_ids:
                continue

            state['missing_frames'] += 1
            if state['missing_frames'] > self.max_missing_frames:
                to_delete.append(tid)
                continue

            # 등속 운동 가정으로 위치 추정
            state['rel_x'] += state['vel_x'] * self.dt
            state['rel_y'] += state['vel_y'] * self.dt
            state['depth'] = max(0.1, state['depth'] + state['vel_y'] * self.dt)
            state['occluded'] = True
            state['predicted_positions'].append((state['rel_x'], state['rel_y']))

            virtual_objects.append({
                'track_id': tid,
                'rel_x': state['rel_x'],
                'rel_y': state['rel_y'],
                'vel_x': state['vel_x'],
                'vel_y': state['vel_y'],
                'depth': state['depth'],
                'occluded': True,
                'missing_frames': state['missing_frames'],
            })

        for tid in to_delete:
            del self._states[tid]

        result = list(current_objects)
        for obj in result:
            obj.setdefault('occluded', False)
            obj.setdefault('missing_frames', 0)
        result.extend(virtual_objects)
        return result

    def reappearance_error(self, track_id: str, actual_x: float, actual_y: float) -> float | None:
        """
        T11: 가림 구간 종료 후 재등장 시 예측 위치와의 오차 반환 (m).
        상태가 없거나 occlusion 이력이 없으면 None.
        """
        state = self._states.get(track_id)
        if state is None or not state['predicted_positions']:
            return None
        last_pred = state['predicted_positions'][-1]
        error = np.sqrt((actual_x - last_pred[0]) ** 2 + (actual_y - last_pred[1]) ** 2)
        return float(error)

    def is_occluded(self, track_id: str) -> bool:
        state = self._states.get(track_id)
        return state is not None and state.get('occluded', False)

    def stats(self) -> dict:
        """현재 occlusion 상태 요약"""
        occluded = [tid for tid, s in self._states.items() if s.get('occluded')]
        return {
            'total_tracked': len(self._states),
            'occluded_count': len(occluded),
            'occluded_ids': occluded,
        }
