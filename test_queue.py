import numpy as np
import threading
import time
from src.pipeline.queue_pipeline import QueuePipeline, FramePacket, ResultPacket


def dummy_worker(pipeline: QueuePipeline):
    """Dummy 모델 — 입력 받아서 고정 TTC 0.5~9.5초 사이 반환."""
    while True:
        packet = pipeline.get_input(timeout=2.0)
        if packet is None:
            break

        # Dummy 추론 (실제 모델 대신 랜덤값)
        ttc = float(np.random.uniform(0.5, 9.5))
        status = "Danger" if ttc <= 1.5 else "Warning" if ttc <= 3.0 else "Safe"

        result = ResultPacket(
            object_id=packet.object_id,
            frame_no=packet.frame_no,
            ttc=round(ttc, 2),
            status=status,
        )
        pipeline.put_result(result)


def test_queue_pipeline():
    pipeline = QueuePipeline(maxsize=50)
    N = 20  # 테스트 패킷 수

    # 워커 스레드 시작
    worker = threading.Thread(target=dummy_worker, args=(pipeline,), daemon=True)
    worker.start()

    # 패킷 삽입
    for i in range(N):
        packet = FramePacket(
            object_id=i % 5,
            frame_no=i,
            pix_data=np.zeros((3, 5), dtype=np.float32),
            real_3d=np.zeros((5, 3), dtype=np.float32),
            images=np.zeros((3, 4, 64, 64), dtype=np.float32),
        )
        ok = pipeline.put(packet)
        assert ok, f"패킷 삽입 실패: frame_no={i}"

    # 결과 수신 및 순서 검증
    results = []
    for _ in range(N):
        result = pipeline.get(timeout=3.0)
        assert result is not None, "결과 수신 타임아웃"
        results.append(result)

    # 입출력 순서 일치 확인 (FIFO)
    in_frames  = list(range(N))
    out_frames = sorted([r.frame_no for r in results])
    assert in_frames == out_frames, f"순서 불일치: {out_frames}"

    stats = pipeline.stats()
    print(f"✅ 총 입력: {stats['total_in']} / 총 출력: {stats['total_out']}")
    print(f"✅ 입출력 순서 일치 확인 완료 (FIFO)")
    print(f"✅ 샘플 결과: frame={results[0].frame_no} ttc={results[0].ttc}s status={results[0].status}")
    print("✅ Dummy 스트리밍 파이프라인 검증 완료")


if __name__ == "__main__":
    test_queue_pipeline()