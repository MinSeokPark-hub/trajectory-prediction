# src/utils/sat_ttc.py
"""
SAT(분리축 정리) 기반 TTC 계산 — 논문 3.2.1절.

두 직사각형(자차 박스, 객체 박스)이 현재 속도로 직선 등속 운동한다고
가정했을 때, 바닥면이 처음 겹치는 시각을 TTC로 정의한다.

구현 방식(검증 용이한 버전)
------------------------------------------------------------------
- 상대 속도 rel_v = obj_v - ego_v (모두 자차 로컬 프레임, ego_yaw=0 가정)
- 각 분리축(자차 2 + 객체 2)에서 두 박스의 1D 투영이 겹치는 시간 구간을
  [lo/v, hi/v] 로 직접 계산 (lo=a_min-b_max, hi=a_max-b_min).
- 4개 축 구간의 교집합이 시작되는 시각이 TTC.
- 지금 이미 겹쳐 있으면 0.0(접촉), 끝까지 안 겹치면 TTC_MAX.
"""
import numpy as np

# 자차 박스 설정 (논문 설정 C: Renault Zoe 제원)
EGO_W = 1.730      # 폭 (m)
EGO_L = 4.084      # 길이 (m)
EGO_OFFSET = 1.4   # 박스 중심을 기준점보다 앞으로 (m)
TTC_MAX = 10.0     # 충돌 없으면 10초


def _get_rect_axes(yaw: float) -> np.ndarray:
    """직사각형의 두 단위 분리축 (length축, width축)."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, s], [-s, c]])


def _project_rect(center: np.ndarray, half_l: float, half_w: float,
                  yaw: float, axis: np.ndarray) -> tuple:
    """회전 직사각형을 axis에 투영한 [min, max]."""
    axes = _get_rect_axes(yaw)
    corners = np.array([
        center + axes[0] * half_l + axes[1] * half_w,
        center + axes[0] * half_l - axes[1] * half_w,
        center - axes[0] * half_l + axes[1] * half_w,
        center - axes[0] * half_l - axes[1] * half_w,
    ])
    proj = corners @ axis
    return float(proj.min()), float(proj.max())


def _axis_overlap_time(a_min, a_max, b_min, b_max, v):
    """
    축 위에서 box1=[a_min,a_max](고정), box2=[b_min,b_max]가 속도 v로
    움직일 때, 두 투영이 겹치는 시간 구간 [t_enter, t_exit] 반환.

    box2(t) = [b_min + v t, b_max + v t].
    겹침 조건:  a_min - b_max <= v t <= a_max - b_min.
    """
    lo = a_min - b_max
    hi = a_max - b_min          # 항상 lo <= hi
    if abs(v) < 1e-12:
        # 상대 운동 없음: 지금 겹쳐 있으면 영원히, 아니면 전혀
        if lo <= 0.0 <= hi:
            return -np.inf, np.inf
        return np.inf, -np.inf   # 빈 구간
    t1, t2 = lo / v, hi / v
    return (t1, t2) if t1 <= t2 else (t2, t1)


def compute_ttc_sat(
    obj_x: float, obj_y: float, obj_yaw: float,
    obj_w: float, obj_l: float,
    obj_vx: float, obj_vy: float,
    ego_vx: float, ego_vy: float,
    ego_yaw: float = 0.0,
) -> float:
    """
    SAT 기반 TTC (초). 입력은 모두 자차 로컬 프레임, ego_yaw=0 가정.

    obj_x, obj_y   : 객체 중심 (자차 기준 상대 좌표, m)
    obj_yaw        : 객체 heading (rad)
    obj_w, obj_l   : 객체 폭·길이 (m)
    obj_vx, obj_vy : 객체 속도 (자차 로컬 프레임, m/s)
    ego_vx, ego_vy : 자차 속도 (자차 로컬 프레임, m/s)

    반환: TTC. 이미 접촉이면 0.0, 충돌 없으면 TTC_MAX.
    """
    ego_center = np.array([EGO_OFFSET * np.cos(ego_yaw),
                           EGO_OFFSET * np.sin(ego_yaw)])
    obj_center = np.array([obj_x, obj_y])

    # 상대 속도 (객체 - 자차)
    rel_v = np.array([obj_vx - ego_vx, obj_vy - ego_vy])

    # 분리축: 자차 2축 + 객체 2축
    axes = np.vstack([_get_rect_axes(ego_yaw), _get_rect_axes(obj_yaw)])

    t_enter_g, t_exit_g = -np.inf, np.inf
    for axis in axes:
        a_min, a_max = _project_rect(ego_center, EGO_L / 2, EGO_W / 2,
                                     ego_yaw, axis)
        b_min, b_max = _project_rect(obj_center, obj_l / 2, obj_w / 2,
                                     obj_yaw, axis)
        v = float(rel_v @ axis)
        te, tx = _axis_overlap_time(a_min, a_max, b_min, b_max, v)
        if te > t_enter_g:
            t_enter_g = te
        if tx < t_exit_g:
            t_exit_g = tx
        if t_enter_g > t_exit_g:      # 교집합 없음 → 충돌 안 함
            return TTC_MAX

    # 겹침 구간 [t_enter_g, t_exit_g]
    if t_exit_g < 0.0:                # 겹침이 과거에만 → 충돌 안 함
        return TTC_MAX
    if t_enter_g <= 0.0:              # 지금 이미 겹침 → 접촉
        return 0.0
    return float(min(t_enter_g, TTC_MAX))