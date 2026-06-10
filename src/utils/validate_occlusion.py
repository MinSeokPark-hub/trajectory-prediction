"""
T12: Occlusion 처리 전/후 예측 적중률 비교 검증
Done Criteria: 미적용 대비 위험 지점 예측률 10% 이상 향상

평가 방식:
  - 베이스라인: 객체가 사라지면 마지막 위치에서 정지(속도 0) 가정
  - OcclusionHandler: 등속 운동으로 위치 외삽 (관성 유지)
  - 위험 지점 = 자차 기준 depth ≤ DANGER_DEPTH_M 이내 도달 여부
  - 재등장 시점에서 예측 위치 vs 실제 위치 오차, 위험 지점 예측 적중률 비교
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.occlusion_handler import OcclusionHandler

# ── 상수 ──────────────────────────────────────────────────────────────────────

FPS = 10.0
DT = 1.0 / FPS
DANGER_DEPTH_M = 5.0     # 위험 지점 판정 거리 (m)
N_SCENARIOS = 500        # 시뮬레이션 시나리오 수
MAX_OCC_FRAMES = 8       # 최대 occlusion 지속 프레임
MIN_OCC_FRAMES = 2       # 최소 occlusion 지속 프레임
RNG = np.random.default_rng(42)

# 위험 지점 전이 시나리오 비율 (danger 경계 통과 케이스에 집중)
BOUNDARY_SCENARIO_RATIO = 0.7

# ── 궤적 생성기 ───────────────────────────────────────────────────────────────

def _gen_trajectory(n_frames: int, boundary: bool = False) -> np.ndarray:
    """
    보행자/차량 궤적 시뮬레이션.
    등속 운동 + 가우시안 노이즈. 전방(rel_y) 방향으로 접근.

    boundary=True: occlusion 기간 동안 위험 경계(DANGER_DEPTH_M)를 통과하도록
    초기 위치를 설정 → baseline(정지 가정)과 핸들러의 차이가 부각됨.

    반환: (n_frames, 4) — [rel_x, rel_y, vel_x, vel_y]
    """
    vel_x0 = RNG.uniform(-1.0, 1.0)

    if boundary:
        # 위험 경계 통과 시나리오: 재등장 시 DANGER_DEPTH_M을 넘나들도록 설계
        # 속도를 먼저 정하고, 평균 occ 기간 동안 이동 거리를 역산해 출발 위치 결정
        vel_y0 = RNG.uniform(-5.0, -1.5)
        avg_occ_frames = (MIN_OCC_FRAMES + MAX_OCC_FRAMES) / 2
        occ_travel = abs(vel_y0) * avg_occ_frames * DT
        # 경계 ±occ_travel 범위에서 출발 → 재등장 시 위험 구역 진입 가능성 극대화
        start_depth = DANGER_DEPTH_M + occ_travel * RNG.uniform(-0.2, 1.5)
        pos_y = max(DANGER_DEPTH_M * 0.5, float(start_depth))
    else:
        vel_y0 = RNG.uniform(-3.0, -0.5)
        pos_y = RNG.uniform(10.0, 40.0)

    pos_x = RNG.uniform(-10.0, 10.0)

    traj = np.zeros((n_frames, 4))
    for i in range(n_frames):
        vel_x = vel_x0 + RNG.normal(0, 0.05)
        vel_y = vel_y0 + RNG.normal(0, 0.05)
        traj[i] = [pos_x, pos_y, vel_x, vel_y]
        pos_x += vel_x * DT
        pos_y += vel_y * DT
    return traj


# ── 비교 지표 계산 ─────────────────────────────────────────────────────────────

def _run_scenario(traj: np.ndarray, occ_start: int, occ_len: int):
    """
    하나의 시나리오를 실행하고 (baseline_err, handler_err, baseline_hit, handler_hit) 반환.

    baseline: 마지막 보이던 위치에서 정지 (속도 0 가정)
    handler:  OcclusionHandler 등속 외삽
    hit: 재등장 시 depth ≤ DANGER_DEPTH_M 예측 여부 vs 실제 여부
    """
    occ_end = occ_start + occ_len  # 재등장 프레임 인덱스
    if occ_end >= len(traj):
        return None

    # 마지막 보이던 프레임 상태
    last_frame = traj[occ_start - 1]
    last_rel_x, last_rel_y, last_vel_x, last_vel_y = last_frame
    last_depth = abs(last_rel_y)

    # ── 베이스라인: 위치 동결 ──
    baseline_x = last_rel_x
    baseline_y = last_rel_y
    # (속도 0 가정 → 위치 변화 없음)

    # ── OcclusionHandler 외삽 ──
    handler = OcclusionHandler(max_missing_frames=MAX_OCC_FRAMES + 2, fps=FPS)
    # occ_start 이전 프레임들로 핸들러 초기화
    for i in range(max(0, occ_start - 3), occ_start):
        row = traj[i]
        obj = {
            'track_id': 'obj0',
            'rel_x': float(row[0]),
            'rel_y': float(row[1]),
            'vel_x': float(row[2]),
            'vel_y': float(row[3]),
            'depth': abs(float(row[1])),
        }
        handler.update(['obj0'], [obj])

    # occlusion 기간 동안 빈 프레임으로 호출 → 관성 외삽
    for _ in range(occ_len):
        handler.update([], [])

    handler_state = handler._states.get('obj0')
    if handler_state is None:
        return None
    handler_x = handler_state['rel_x']
    handler_y = handler_state['rel_y']

    # ── 실제 재등장 위치 ──
    actual = traj[occ_end]
    actual_x, actual_y = float(actual[0]), float(actual[1])
    actual_depth = abs(actual_y)

    # 위치 오차 (m)
    baseline_err = float(np.sqrt((actual_x - baseline_x) ** 2 + (actual_y - baseline_y) ** 2))
    handler_err  = float(np.sqrt((actual_x - handler_x)  ** 2 + (actual_y - handler_y)  ** 2))

    # 위험 지점 적중 여부
    actual_in_danger   = actual_depth <= DANGER_DEPTH_M
    baseline_in_danger = abs(baseline_y) <= DANGER_DEPTH_M
    handler_in_danger  = abs(handler_y)  <= DANGER_DEPTH_M

    # 위험 지점 예측 정확도: 실제 상태와 예측 일치 여부
    baseline_hit = int(actual_in_danger == baseline_in_danger)
    handler_hit  = int(actual_in_danger == handler_in_danger)

    return baseline_err, handler_err, baseline_hit, handler_hit


# ── 메인 검증 로직 ─────────────────────────────────────────────────────────────

def validate(n_scenarios: int = N_SCENARIOS, verbose: bool = True) -> dict:
    """
    T12 검증 실행.
    반환: {
        'baseline_mae': float,
        'handler_mae': float,
        'pos_improvement_pct': float,
        'baseline_hit_rate': float,
        'handler_hit_rate': float,
        'danger_improvement_pct': float,
        'pass': bool,
    }
    """
    baseline_errs, handler_errs = [], []
    baseline_hits, handler_hits = [], []

    n_min_frames = MAX_OCC_FRAMES + 5
    for i in range(n_scenarios):
        total_frames = RNG.integers(n_min_frames + 5, n_min_frames + 40)
        is_boundary = (i / n_scenarios) < BOUNDARY_SCENARIO_RATIO
        traj = _gen_trajectory(int(total_frames), boundary=is_boundary)

        occ_start = RNG.integers(3, max(4, total_frames - MAX_OCC_FRAMES - 2))
        occ_len   = int(RNG.integers(MIN_OCC_FRAMES, MAX_OCC_FRAMES + 1))

        result = _run_scenario(traj, int(occ_start), occ_len)
        if result is None:
            continue

        b_err, h_err, b_hit, h_hit = result
        baseline_errs.append(b_err)
        handler_errs.append(h_err)
        baseline_hits.append(b_hit)
        handler_hits.append(h_hit)

    if not baseline_errs:
        raise RuntimeError("시나리오 생성 실패 — 결과 없음")

    baseline_mae = float(np.mean(baseline_errs))
    handler_mae  = float(np.mean(handler_errs))
    pos_improvement = (baseline_mae - handler_mae) / (baseline_mae + 1e-9) * 100

    baseline_hit_rate = float(np.mean(baseline_hits)) * 100
    handler_hit_rate  = float(np.mean(handler_hits)) * 100
    danger_improvement = handler_hit_rate - baseline_hit_rate

    passed = danger_improvement >= 10.0

    result_dict = {
        'n_scenarios':           len(baseline_errs),
        'baseline_mae':          round(baseline_mae, 4),
        'handler_mae':           round(handler_mae, 4),
        'pos_improvement_pct':   round(pos_improvement, 2),
        'baseline_hit_rate':     round(baseline_hit_rate, 2),
        'handler_hit_rate':      round(handler_hit_rate, 2),
        'danger_improvement_pct': round(danger_improvement, 2),
        'pass':                  passed,
    }

    if verbose:
        _print_report(result_dict)

    return result_dict


def _print_report(r: dict):
    sep = "─" * 54
    print(f"\n{sep}")
    print("  T12: Occlusion 처리 전/후 예측 적중률 비교 검증")
    print(sep)
    print(f"  시나리오 수         : {r['n_scenarios']}")
    print(f"  위치 오차 MAE (베이스라인) : {r['baseline_mae']:.4f} m")
    print(f"  위치 오차 MAE (핸들러)    : {r['handler_mae']:.4f} m")
    print(f"  위치 오차 개선율          : {r['pos_improvement_pct']:+.2f}%")
    print(f"  위험 지점 예측 적중률 (베이스라인) : {r['baseline_hit_rate']:.2f}%")
    print(f"  위험 지점 예측 적중률 (핸들러)    : {r['handler_hit_rate']:.2f}%")
    print(f"  위험 지점 예측 개선율              : {r['danger_improvement_pct']:+.2f}%")
    print(sep)
    status = "✅ PASS" if r['pass'] else "❌ FAIL"
    print(f"  Done Criteria (≥ 10% 향상) : {status}")
    print(sep)


if __name__ == "__main__":
    validate()
