# src/utils/sat_ttc.py
import numpy as np
from typing import Optional


# 자차 박스 설정 (논문 설정 C: Renault Zoe 제원)
EGO_W = 1.730   # 폭 (m)
EGO_L = 4.084   # 길이 (m)
EGO_OFFSET = 1.4  # 중심 오프셋 (앞으로)
TTC_MAX = 10.0   # 충돌 없으면 10초


def _get_rect_axes(yaw: float) -> np.ndarray:
    """직사각형의 두 분리축 (회전 반영)"""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, s], [-s, c]])  # (2, 2)


def _project_rect(center: np.ndarray, half_l: float, half_w: float,
                  yaw: float, axis: np.ndarray) -> tuple:
    """직사각형을 축에 투영한 [min, max] 반환"""
    axes = _get_rect_axes(yaw)
    corners = np.array([
        center + axes[0] * half_l + axes[1] * half_w,
        center + axes[0] * half_l - axes[1] * half_w,
        center - axes[0] * half_l + axes[1] * half_w,
        center - axes[0] * half_l - axes[1] * half_w,
    ])  # (4, 2)
    proj = corners @ axis
    return proj.min(), proj.max()


def _overlap_interval(c1: np.ndarray, hl1: float, hw1: float, yaw1: float,
                      c2: np.ndarray, hl2: float, hw2: float, yaw2: float,
                      vc: np.ndarray, axis: np.ndarray) -> tuple:
    """
    한 축에서 두 직사각형의 투영이 겹치는 시간 구간 [t_enter, t_exit] 반환.
    vc = 상대 속도 벡터 (obj - ego)
    """
    min1, max1 = _project_rect(c1, hl1, hw1, yaw1, axis)
    min2, max2 = _project_rect(c2, hl2, hw2, yaw2, axis)
    v = float(vc @ axis)  # 상대속도의 축 방향 성분

    # 이미 겹쳐 있으면 [0, inf]
    if min2 <= max1 and min1 <= max2:
        if abs(v) < 1e-9:
            return 0.0, np.inf
        t1 = (max1 - min2) / v if v != 0 else np.inf
        t2 = (min1 - max2) / v if v != 0 else np.inf
        return min(t1, t2), max(t1, t2)

    # 겹치지 않음 — 언제 겹치기 시작하고 끝나는지
    if abs(v) < 1e-9:
        return np.inf, np.inf  # 평행, 안 만남

    # gap 방향에 따라
    if max1 < min2:
        t_enter = (min2 - max1) / v
        t_exit  = (max2 - min1) / v
    else:
        t_enter = (max1 - min2) / (-v)
        t_exit  = (min1 - max2) / (-v)

    if t_enter > t_exit:
        t_enter, t_exit = t_exit, t_enter

    return t_enter, t_exit


def compute_ttc_sat(
    obj_x: float, obj_y: float, obj_yaw: float,
    obj_w: float, obj_l: float,
    obj_vx: float, obj_vy: float,
    ego_vx: float, ego_vy: float,
    ego_yaw: float = 0.0,
) -> float:
    """
    SAT 기반 TTC 계산 (논문 3.2.1절).

    모든 좌표는 자차 기준 로컬 프레임 (ego 중심이 원점).
    ego_yaw=0 가정 (자차는 항상 전방이 +x).

    Parameters
    ----------
    obj_x, obj_y : 객체 중심 (자차 기준 상대 좌표, m)
    obj_yaw      : 객체 heading (rad, 전방 +x 기준)
    obj_w, obj_l : 객체 폭·길이 (m)
    obj_vx, obj_vy : 객체 속도 (월드 프레임, m/s)
    ego_vx, ego_vy : 자차 속도 (월드 프레임, m/s)
    ego_yaw      : 자차 heading (rad)

    Returns
    -------
    TTC (초). 충돌 없으면 TTC_MAX(10.0).
    """
    # 자차 박스 중심 (오프셋 적용)
    ego_cx = EGO_OFFSET * np.cos(ego_yaw)
    ego_cy = EGO_OFFSET * np.sin(ego_yaw)
    ego_center = np.array([ego_cx, ego_cy])

    obj_center = np.array([obj_x, obj_y])

    # 상대 속도 (obj - ego)
    rel_vx = obj_vx - ego_vx
    rel_vy = obj_vy - ego_vy
    rel_v  = np.array([rel_vx, rel_vy])

    # 이미 겹쳐있으면 TTC = 0 (논문에서 7개 제외한 케이스)
    # → 여기서는 그냥 0 반환
    if np.linalg.norm(rel_v) < 1e-6:
        return TTC_MAX

    # 4개 분리축: 자차 2축 + 객체 2축
    ego_axes = _get_rect_axes(ego_yaw)    # (2, 2)
    obj_axes = _get_rect_axes(obj_yaw)    # (2, 2)
    axes = np.vstack([ego_axes, obj_axes])  # (4, 2)

    t_enter_global = 0.0
    t_exit_global  = np.inf

    for axis in axes:
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        t_en, t_ex = _overlap_interval(
            ego_center, EGO_L / 2, EGO_W / 2, ego_yaw,
            obj_center, obj_l / 2, obj_w / 2, obj_yaw,
            rel_v, axis
        )
        if t_en > t_ex or t_ex < 0:
            # 이 축에서 절대 안 만남
            return TTC_MAX

        t_enter_global = max(t_enter_global, t_en)
        t_exit_global  = min(t_exit_global,  t_ex)

    if t_enter_global > t_exit_global or t_enter_global < 0:
        return TTC_MAX

    ttc = float(t_enter_global)
    return min(ttc, TTC_MAX)