import threading
import queue
import time
from typing import Callable


class AsyncPredictionWorker:
    """
    US-15: 예측 좌표 비동기 산출 엔진

    T16: 이전 연산이 끝나기 전에 다음 연산을 독립적으로 시작
    T17: 객체별 결과 큐 순서 보장
    T18: 30FPS 이상 처리 성능 측정
    """

    def __init__(self, predict_fn: Callable, n_workers: int = 2):
        """
        predict_fn: InferenceEngine.predict_scene 같은 callable
                    signature: predict_fn(objects, fps) → dict
        """
        self._predict_fn = predict_fn
        self._input_queue: queue.Queue = queue.Queue()
        self._result_queues: dict[int, queue.Queue] = {}
        self._lock = threading.Lock()
        self._frame_counter = 0
        self._workers: list[threading.Thread] = []
        self._running = False

        # 성능 측정용
        self._processed_frames = 0
        self._start_time: float | None = None

        for _ in range(n_workers):
            t = threading.Thread(target=self._worker_loop, daemon=True)
            self._workers.append(t)

    # ── 공개 API ──────────────────────────────────────────────────────────────

    def start(self):
        self._running = True
        self._start_time = time.time()
        for t in self._workers:
            t.start()

    def stop(self):
        self._running = False
        # 워커 종료 시그널
        for _ in self._workers:
            self._input_queue.put(None)
        for t in self._workers:
            t.join(timeout=2.0)

    def submit(self, objects: list[dict], fps: float = 10.0) -> int:
        """
        T16: 비동기 제출 — 이전 연산 종료 전에 다음 연산 독립 시작 가능.
        반환값: frame_id (결과 수신 시 사용)
        """
        with self._lock:
            frame_id = self._frame_counter
            self._frame_counter += 1

        result_q: queue.Queue = queue.Queue(maxsize=1)
        with self._lock:
            self._result_queues[frame_id] = result_q

        self._input_queue.put((frame_id, objects, fps))
        return frame_id

    def get_result(self, frame_id: int, timeout: float = 2.0) -> dict | None:
        """
        T17: 해당 frame_id의 결과를 순서 보장하며 반환.
        timeout 내에 결과 없으면 None.
        """
        with self._lock:
            result_q = self._result_queues.get(frame_id)
        if result_q is None:
            return None
        try:
            result = result_q.get(timeout=timeout)
            with self._lock:
                self._result_queues.pop(frame_id, None)
            return result
        except queue.Empty:
            return None

    def fps_stats(self) -> dict:
        """T18: 처리 성능 측정"""
        if self._start_time is None or self._processed_frames == 0:
            return {'fps': 0.0, 'processed': 0}
        elapsed = time.time() - self._start_time
        fps = self._processed_frames / elapsed if elapsed > 0 else 0.0
        return {
            'fps': round(fps, 1),
            'processed': self._processed_frames,
            'elapsed_s': round(elapsed, 2),
        }

    # ── 내부 워커 ──────────────────────────────────────────────────────────────

    def _worker_loop(self):
        while self._running:
            try:
                item = self._input_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if item is None:
                break

            frame_id, objects, fps = item
            t0 = time.time()

            try:
                result = self._predict_fn(objects, fps)
            except Exception as e:
                result = {'error': str(e), 'social': {}, 'predictions': [],
                          'ade_no_social': 0.0, 'ade_social': 0.0}

            result['latency_ms'] = round((time.time() - t0) * 1000, 1)

            with self._lock:
                result_q = self._result_queues.get(frame_id)
                self._processed_frames += 1

            if result_q is not None:
                try:
                    result_q.put_nowait(result)
                except queue.Full:
                    pass
