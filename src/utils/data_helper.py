import os
import pandas as pd
from utils.kitti_parser import KittiParser

def load_all_kitti_data(label_dir):
    parser = KittiParser(fps=10)
    all_sequences = []
    
    # 1. 폴더 내 모든 .txt 파일 목록 가져오기
    label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.txt')])
    print(f"📊 총 {len(label_files)}개의 시퀀스 파일을 발견했습니다.")

    for file_name in label_files:
        file_path = os.path.join(label_dir, file_name)
        
        # 2. 개별 파일 파싱 및 TTC 계산
        raw_df = parser.parse_label(file_path)
        
        if raw_df.empty:
            continue
            
        processed_df = parser.calculate_gt_ttc(raw_df)
        
        # 시퀀스 간 구분을 위해 파일명을 prefix로 붙인 새로운 track_id 생성[cite: 1]
        processed_df['track_id'] = file_name.replace('.txt', '') + "_" + processed_df['track_id'].astype(str)
        
        all_sequences.append(processed_df)

    # 3. 모든 데이터프레임 하나로 합치기[cite: 1]
    final_df = pd.concat(all_sequences, ignore_index=True)
    print(f"✅ 통합 완료: 총 {len(final_df)}개의 보행자 데이터 포인트를 확보했습니다.")
    
    return final_df