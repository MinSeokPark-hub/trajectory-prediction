from src.utils.nuscenes_parser import NuScenesParser

DATAROOT = '/workspace/minseok_park/data/nuscenes/v1.0-mini'

parser = NuScenesParser(dataroot=DATAROOT, version='v1.0-mini', fps=2.0)
df = parser.load()

print(f"컬럼: {df.columns.tolist()}")
print(f"shape: {df.shape}")
print(f"타입 종류: {df['type'].unique()}")
print(f"scene 종류: {df['scene'].unique()}")
print(f"\nego_pose 샘플:")
print(df[['frame', 'track_id', 'type', 'depth', 'ego_x', 'ego_y', 'gt_ttc']].head(10))
print("✅ NuScenesParser 동작 확인 완료")