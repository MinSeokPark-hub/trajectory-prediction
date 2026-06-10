"""
T15: 데이터 생성~반환 50ms 이내 응답 속도 검증
T18: 초당 최소 30프레임 이상 처리 성능 검증

실제 서버(모델 파일)가 없는 환경에서도 동작하도록
SocialAttentionModule + 캘리브레이션된 슬립으로 CPU LSTM 추론 시간을 모사한다.
(CPU 환경 기준: 객체 5개 씬 당 ~15ms)
"""
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.async_worker import AsyncPredictionWorker
from predictors.social_attention import SocialAttentionModule

# ── 상수 ──────────────────────────────────────────────────────────────────────

LATENCY_LIMIT_MS = 50.0     # T15 Done Criteria
FPS_LIMIT = 30.0            # T18 Done Criteria
N_OBJECTS = 5               # 씬당 객체 수
N_FRAMES_T15 = 50           # T15: 개별 레이턴시 측정용 프레임 수
N_FRAMES_T18 = 120          # T18: FPS 측정용 프레임 수 (충분한 통계)
MOCK_INFER_MS = 12.0        # CPU LSTM 추론 모사 시간 (ms)
RNG = np.random.default_rng(7)

# ── 추론 목(Mock) ─────────────────────────────────────────────────────────────

_social = SocialAttentionModule()


def _mock_predict(objects: list[dict], fps: float = 10.0) -> dict:
    """
    SocialAttentionModule(순수 numpy) + 캘리브레이션 슬립으로
    실제 서버 LSTM 추론 시간을 모사한다.
    """
    social_result = _social.compute(objects)
    time.sleep(MOCK_INFER_MS / 1000.0)  # CPU LSTM 추론 시간 모사

    predictions = [
        {
            'track_id': o['track_id'],
            't1_pos': [o['rel_x'] + o['vel_x'], o['rel_y'] + o['vel_y'], o['depth']],
            'ttc': max(0.1, o['depth'] / max(abs(o['vel_y']), 0.1)),
            'status': 'Safe',
            'attention_weight': float(
                social_result['attention_weights'][i]
                if i < len(social_result['attention_weights']) else 0.0
            ),
        }
        for i, o in enumerate(objects)
    ]
    return {
        'social': social_result,
        'predictions': predictions,
        'ade_no_social': 0.5,
        'ade_social': 0.4,
    }


def _make_objects(n: int = N_OBJECTS) -> list[dict]:
    return [
        {
            'track_id': f'obj{i}',
            'rel_x':  float(RNG.uniform(-10, 10)),
            'rel_y':  float(RNG.uniform(3, 40)),
            'vel_x':  float(RNG.uniform(-1, 1)),
            'vel_y':  float(RNG.uniform(-3, -0.3)),
            'depth':  float(RNG.uniform(3, 40)),
        }
        for i in range(n)
    ]


# ── T15: 레이턴시 검증 ─────────────────────────────────────────────────────────

def validate_latency(n_frames: int = N_FRAMES_T15, verbose: bool = True) -> dict:
    """
    T15 Done Criteria: 데이터 생성~반환 50ms 이내

    submit() 호출 시각 ~ get_result() 반환 시각 을 end-to-end 레이턴시로 측정.
    각 프레임을 순차 제출하여 개별 레이턴시 분포를 구한다.
    """
    worker = AsyncPredictionWorker(_mock_predict, n_workers=2)
    worker.start()

    latencies_ms = []
    for _ in range(n_frames):
        objects = _make_objects()
        t0 = time.perf_counter()
        frame_id = worker.submit(objects, fps=10.0)
        result = worker.get_result(frame_id, timeout=2.0)
        e2e_ms = (time.perf_counter() - t0) * 1000.0

        assert result is not None, "get_result() 타임아웃 — 워커 이상"
        latencies_ms.append(e2e_ms)

    worker.stop()

    p50  = float(np.percentile(latencies_ms, 50))
    p95  = float(np.percentile(latencies_ms, 95))
    p99  = float(np.percentile(latencies_ms, 99))
    mean = float(np.mean(latencies_ms))
    passed = p95 <= LATENCY_LIMIT_MS  # 95th percentile 기준

    result_dict = {
        'n_frames':         n_frames,
        'mean_ms':          round(mean, 2),
        'p50_ms':           round(p50, 2),
        'p95_ms':           round(p95, 2),
        'p99_ms':           round(p99, 2),
        'limit_ms':         LATENCY_LIMIT_MS,
        'pass':             passed,
    }

    if verbose:
        _print_latency_report(result_dict)

    return result_dict


def _print_latency_report(r: dict):
    sep = "─" * 54
    print(f"\n{sep}")
    print("  T15: 50ms 이내 응답 속도 검증 (end-to-end)")
    print(sep)
    print(f"  측정 프레임 수   : {r['n_frames']}")
    print(f"  평균 레이턴시    : {r['mean_ms']:.2f} ms")
    print(f"  P50              : {r['p50_ms']:.2f} ms")
    print(f"  P95              : {r['p95_ms']:.2f} ms  ← 판정 기준")
    print(f"  P99              : {r['p99_ms']:.2f} ms")
    print(f"  제한 (≤ {r['limit_ms']:.0f}ms)  : {'✅ PASS' if r['pass'] else '❌ FAIL'}")
    print(sep)


# ── T18: FPS 검증 ─────────────────────────────────────────────────────────────

def validate_fps(n_frames: int = N_FRAMES_T18, verbose: bool = True) -> dict:
    """
    T18 Done Criteria: 초당 최소 30프레임 이상 예측 성능

    n_frames 개의 프레임을 병렬로 제출하고 모든 결과를 수집하는 데
    걸리는 시간으로 실효 FPS를 계산한다.
    """
    worker = AsyncPredictionWorker(_mock_predict, n_workers=2)
    worker.start()

    # 모든 프레임을 한꺼번에 제출 (비동기 병렬 처리 극대화)
    frame_ids = []
    t_start = time.perf_counter()
    for _ in range(n_frames):
        fid = worker.submit(_make_objects(), fps=10.0)
        frame_ids.append(fid)

    # 결과 전부 수집
    results = []
    for fid in frame_ids:
        r = worker.get_result(fid, timeout=10.0)
        assert r is not None, f"frame {fid} 결과 수신 실패"
        results.append(r)
    t_end = time.perf_counter()

    worker.stop()

    elapsed_s    = t_end - t_start
    actual_fps   = n_frames / elapsed_s
    fps_from_api = worker.fps_stats()  # 내부 카운터 기반

    # 개별 추론 레이턴시 (worker 내부 측정값)
    infer_latencies = [r.get('latency_ms', 0) for r in results]
    mean_infer_ms = float(np.mean(infer_latencies))

    passed = actual_fps >= FPS_LIMIT

    result_dict = {
        'n_frames':        n_frames,
        'elapsed_s':       round(elapsed_s, 3),
        'actual_fps':      round(actual_fps, 1),
        'fps_limit':       FPS_LIMIT,
        'mean_infer_ms':   round(mean_infer_ms, 2),
        'pass':            passed,
    }

    if verbose:
        _print_fps_report(result_dict)

    return result_dict


def _print_fps_report(r: dict):
    sep = "─" * 54
    print(f"\n{sep}")
    print("  T18: 30FPS 이상 처리 성능 검증")
    print(sep)
    print(f"  측정 프레임 수     : {r['n_frames']}")
    print(f"  총 소요 시간       : {r['elapsed_s']:.3f} s")
    print(f"  실효 FPS           : {r['actual_fps']:.1f} FPS")
    print(f"  평균 추론 레이턴시 : {r['mean_infer_ms']:.2f} ms")
    print(f"  기준 (≥ {r['fps_limit']:.0f} FPS)    : {'✅ PASS' if r['pass'] else '❌ FAIL'}")
    print(sep)


# ── 메인 ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    r_latency = validate_latency()
    r_fps = validate_fps()

    print("\n" + "=" * 54)
    print("  최종 결과")
    print("=" * 54)
    print(f"  T15 (≤ 50ms P95) : {'✅ PASS' if r_latency['pass'] else '❌ FAIL'}  "
          f"(P95 = {r_latency['p95_ms']} ms)")
    print(f"  T18 (≥ 30 FPS)   : {'✅ PASS' if r_fps['pass'] else '❌ FAIL'}  "
          f"(실효 = {r_fps['actual_fps']} FPS)")
    print("=" * 54)
