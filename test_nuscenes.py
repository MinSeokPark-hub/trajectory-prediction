from nuscenes.nuscenes import NuScenes

DATAROOT = '/workspace/minseok_park/data/nuscenes/v1.0-mini'

nusc = NuScenes(version='v1.0-mini', dataroot=DATAROOT, verbose=True)

# 기본 정보 출력
print(f"\n총 scene 수: {len(nusc.scene)}")
print(f"총 sample 수: {len(nusc.sample)}")

# 첫 번째 scene 확인
scene = nusc.scene[0]
print(f"\n첫 번째 scene: {scene['name']}")
print(f"설명: {scene['description']}")

# ego_pose (GPS) 확인
ego = nusc.ego_pose[0]
print(f"\nego_pose 샘플:")
print(f"  translation (x,y,z): {ego['translation']}")
print(f"  timestamp: {ego['timestamp']}")

# 카메라 샘플 확인
sample = nusc.sample[0]
cam_token = sample['data']['CAM_FRONT']
cam_data = nusc.get('sample_data', cam_token)
print(f"\n카메라 이미지 경로: {cam_data['filename']}")
print("✅ nuScenes 데이터 로드 완료")