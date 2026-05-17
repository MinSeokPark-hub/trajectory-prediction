import torch
import numpy as np
from utils.kitti_parser import KittiParser
from inference_engine import InferenceEngine

def run_integration_test():
    # 1. 엔진 및 데이터 로더 초기화
    engine = InferenceEngine()
    parser = KittiParser(fps=10)
    
    # 테스트할 시퀀스 파일 선택
    test_file = "/workspace/minseok_park/data/kitti/labels/0000.txt"
    raw_df = parser.parse_label(test_file)
    processed_df = parser.calculate_gt_ttc(raw_df)
    
    print(f"\n🚀 통합 테스트 시작: {test_file}")
    print(f"{'Frame':<8} | {'ID':<4} | {'SimpleTTC':<10} | {'LSTM-TTC':<10} | {'Final':<8} | {'Status':<8}")
    print("-" * 70)

    # 2. 프레임별 시뮬레이션
    # 각 track_id 별로 시퀀스 데이터를 모아서 추론 엔진에 전달합니다.
    for tid in processed_df['track_id'].unique():
        ped_data = processed_df[processed_df['track_id'] == tid]
        
        # LSTM 입력에 필요한 최소 5프레임 이상인 경우만 테스트
        if len(ped_data) < 5:
            continue
            
        # 슬라이딩 윈도우 방식으로 프레임 진행 시뮬레이션
        for i in range(5, len(ped_data)):
            # 최근 5프레임 데이터 추출
            window = ped_data.iloc[i-5:i]
            # [x, y, w, h, depth] 형식으로 데이터 구성
            sequence = window[['x', 'y', 'w', 'h', 'depth']].values
            
            # 통합 엔진 추론
            final_ttc, status = engine.predict(sequence)
            
            # 비교를 위해 개별 모델 값 가져오기 (디버깅용)
            # (InferenceEngine 내부 로직을 잠시 활용)
            frame_idx = window.iloc[-1]['frame']
            
            if i % 10 == 0: # 너무 많이 찍히지 않게 10프레임마다 출력
                print(f"{int(frame_idx):<8} | {tid:<4} | {final_ttc:>.2f}s | {status:<8}")

if __name__ == "__main__":
    run_integration_test()