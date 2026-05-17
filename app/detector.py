import cv2
import torch
import os
from ultralytics import YOLO

def get_pedestrian_tracks(video_path):
    # 1. YOLOv8 모델 로드 (가장 가벼운 nano 버전)
    model = YOLO('yolov8n.pt')
    
    # 2. 영상 열기
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ 영상을 열 수 없습니다. 경로를 확인하세요.")
        return []

    print("🚀 YOLOv8 보행자 탐지 및 좌표 추출 시작...")
    
    frame_results = []
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # 3. YOLO로 사람(class 0)만 탐지 (성능을 위해 5프레임당 1번씩만 확인)
        if frame_count % 5 == 0:
            results = model.predict(frame, classes=[0], verbose=False)
            
            for result in results:
                boxes = result.boxes.xywh.cpu().numpy() # 중심 X, Y, 가로, 세로
                for box in boxes:
                    x, y, w, h = box
                    # 프레임 번호와 중심 좌표(x, y) 저장
                    frame_results.append([frame_count, x, y])
        
        frame_count += 1
        if frame_count > 100: break # 테스트를 위해 100프레임까지만

    cap.release()
    print(f"✅ 추출 완료! 총 {len(frame_results)}개의 보행자 위치를 파악했습니다.")
    return frame_results

if __name__ == "__main__":
    video = "data/test_video.mp4"
    tracks = get_pedestrian_tracks(video)
    
    # 파일로 저장하는 로직 추가
    import numpy as np
    os.makedirs("data", exist_ok=True)
    # [프레임, ID(임시1), X, Y] 형식으로 저장
    save_data = [[t[0], 1, t[1], t[2]] for t in tracks]
    np.savetxt("data/custom_obsmat.txt", save_data, fmt='%d %d %.4f %.4f')
    
    print(f"💾 좌표 데이터가 'data/custom_obsmat.txt'에 저장되었습니다!")