# src/utils/ego_motion.py
"""
논문 3.2.2절 — 자차 운동 정보 추출.
4프레임 × 5특징 = 20-dim 벡터.
특징: [speed, lon_accel, rotation_rate, dx, dy]
"""
import numpy as np
from pyquaternion import Quaternion


def _quat_to_yaw(q: list) -> float:
    """nuScenes quaternion [w,x,y,z] → yaw (rad)"""
    quat = Quaternion(q[0], q[1], q[2], q[3])
    return quat.yaw_pitch_roll[0]


def extract_ego_motion(ego_poses: list, dt: float = 0.5) -> np.ndarray:
    """
    Parameters
    ----------
    ego_poses : list of dict, 길이 5 (현재 + 이전 4프레임)
        각 dict: {'translation': [x,y,z], 'rotation': [w,x,y,z]}
        순서: [t-4, t-3, t-2, t-1, t] (과거→현재)
    dt : 프레임 간격 (nuScenes CAM_FRONT ≈ 0.5s)

    Returns
    -------
    feat : np.ndarray (20,)  — 4프레임 × 5특징
    """
    assert len(ego_poses) == 5, "ego_poses must have 5 entries (t-4 to t)"

    features = []
    for i in range(1, 5):   # i=1..4 → 4개 프레임 차이
        prev = ego_poses[i - 1]
        curr = ego_poses[i]

        px, py = prev['translation'][0], prev['translation'][1]
        cx, cy = curr['translation'][0], curr['translation'][1]

        dx = cx - px
        dy = cy - py
        speed = np.sqrt(dx**2 + dy**2) / dt

        yaw_prev = _quat_to_yaw(prev['rotation'])
        yaw_curr = _quat_to_yaw(curr['rotation'])
        dyaw = yaw_curr - yaw_prev
        # wrap to [-pi, pi]
        dyaw = (dyaw + np.pi) % (2 * np.pi) - np.pi
        rotation_rate = dyaw / dt

        # 종방향 가속도: 속도 변화 (이전 프레임 속도 필요하면 근사)
        if i == 1:
            lon_accel = 0.0
        else:
            pprev = ego_poses[i - 2]
            ppx, ppy = pprev['translation'][0], pprev['translation'][1]
            prev_speed = np.sqrt((px-ppx)**2 + (py-ppy)**2) / dt
            lon_accel = (speed - prev_speed) / dt

        features.extend([speed, lon_accel, rotation_rate, dx, dy])

    return np.array(features, dtype=np.float32)  # (20,)


def extract_ego_motion_from_sample(nusc, sample_token: str) -> np.ndarray:
    """
    nuScenes API를 사용해 sample_token 기준 4프레임 이전까지 ego_pose 수집.
    부족하면 가장 오래된 pose로 패딩.
    """
    sample = nusc.get('sample', sample_token)
    cam_data = nusc.get('sample_data', sample['data']['CAM_FRONT'])

    poses = []
    curr = cam_data
    for _ in range(5):
        ep = nusc.get('ego_pose', curr['ego_pose_token'])
        poses.append(ep)
        if curr['prev'] == '':
            break
        curr = nusc.get('sample_data', curr['prev'])

    # 오래된 순으로 뒤집기 (과거→현재)
    poses = poses[::-1]

    # 부족하면 첫 번째로 패딩
    while len(poses) < 5:
        poses.insert(0, poses[0])

    return extract_ego_motion(poses[:5])
