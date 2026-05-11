import queue
import threading
import logging
from dataclasses import dataclass
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FramePacket:
    """큐를 통해 전달되는 데이터 단위."""
    object_id: int
    frame_no:  int
    pix_data:  np.ndarray   # (5,) or (N, 5) — SimpleTTC용
    real_3d:   np.ndarray   # (N, 3)          — LSTM용
    images:    np.ndarray   # (C, T, H, W)    — 3D CNN용


@dataclass
class ResultPacket:
    """추론 결과 단위."""
    object_id: int
    frame_no:  int
    ttc:       float
    status:    str           # Safe / Warning / Danger


class QueuePipeline:
    """
    입력 큐 → 추론 → 출력 큐 FIFO 파이프라인.

    사용법:
        pipeline = QueuePipeline(maxsize=100)
        pipeline.put(packet)        # 입력 데이터 삽입
        result = pipeline.get()     # 추론 결과 수신
    """

    def __init__(self, maxsize: int = 100):
        self._in_queue:  queue.Queue = queue.Queue(maxsize=maxsize)
        self._out_queue: queue.Queue = queue.Queue(maxsize=maxsize)
        self._lock = threading.Lock()
        self._in_count  = 0
        self._out_count = 0

    # ── 입력 ──────────────────────────────────────────────────

    def put(self, packet: FramePacket, timeout: float = 1.0) -> bool:
        """
        입력 큐에 FramePacket 삽입.
        큐가 꽉 찼을 경우 timeout 후 False 반환 (블로킹 없음).
        """
        try:
            self._in_queue.put(packet, timeout=timeout)
            with self._lock:
                self._in_count += 1
            return True
        except queue.Full:
            logger.warning(f"입력 큐 포화 — packet 드롭 (object_id={packet.object_id})")
            return False

    def get_input(self, timeout: float = 1.0) -> Optional[FramePacket]:
        """추론 워커가 입력 큐에서 패킷을 꺼낼 때 사용."""
        try:
            return self._in_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    # ── 출력 ──────────────────────────────────────────────────

    def put_result(self, result: ResultPacket, timeout: float = 1.0) -> bool:
        """추론 워커가 결과를 출력 큐에 삽입."""
        try:
            self._out_queue.put(result, timeout=timeout)
            with self._lock:
                self._out_count += 1
            return True
        except queue.Full:
            logger.warning(f"출력 큐 포화 — result 드롭 (object_id={result.object_id})")
            return False

    def get(self, timeout: float = 1.0) -> Optional[ResultPacket]:
        """클라이언트가 추론 결과를 수신할 때 사용."""
        try:
            return self._out_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    # ── 상태 ──────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "in_queue_size":  self._in_queue.qsize(),
            "out_queue_size": self._out_queue.qsize(),
            "total_in":       self._in_count,
            "total_out":      self._out_count,
        }

    def empty(self) -> bool:
        return self._in_queue.empty()