import os  # <--- 이 줄이 빠져서 에러가 났습니다!
from kitti_parser import KittiParser

# 1. 객체 생성 (KITTI는 보통 10Hz이므로 fps=10)
parser = KittiParser(fps=10)

# 2. 예시 파일 로드
# 데이터 업로드 여부에 따라 경로를 확인해 주세요.
label_path = "/workspace/minseok_park/data/kitti/labels/0000.txt"

if os.path.exists(label_path):
    # 데이터 파싱 및 정답 TTC 계산[cite: 1]
    raw_ped_data = parser.parse_label(label_path)
    final_data = parser.calculate_gt_ttc(raw_ped_data)
    
    print("--- 전처리된 데이터 샘플 ---")
    print(final_data[['frame', 'track_id', 'depth', 'velocity', 'gt_ttc']].head())
else:
    print(f"❌ 파일을 찾을 수 없습니다: {label_path}")
    print("데이터를 'data/kitti/labels/' 폴더에 업로드했는지 확인해 주세요.[cite: 1]")