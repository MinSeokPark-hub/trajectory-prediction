import cv2
import numpy as np
import torch

def extract_frames(video_path, num_frames=16, resize=(112, 112)):
    """
    비디오에서 지정된 개수(num_frames)만큼 연속된 프레임을 추출하고 전처리하여 
    3D CNN 모델에 들어갈 수 있는 PyTorch 텐서 형태로 반환합니다.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret: # 영상이 끝나면 종료
            break
        
        # 1. BGR -> RGB 색상 변환 (OpenCV는 기본이 BGR이므로 변환 필수)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 2. 이미지 크기 조절 (연산 속도를 위해 112x112 등으로 축소)
        frame = cv2.resize(frame, resize)
        
        # 3. 정규화 (픽셀값 0~255를 0.0~1.0 사이의 소수로 변환하여 학습 효율 상승)
        frame = frame.astype(np.float32) / 255.0
        
        frames.append(frame)
        
    cap.release()
    
    # 프레임이 부족한 경우의 예외 처리
    if len(frames) < num_frames:
        print(f"경고: 영상이 너무 짧습니다. (추출된 프레임: {len(frames)}/{num_frames})")
        return None
        
    # 4. 리스트를 Numpy 배열로 변환: 현재 형태 -> (프레임수, 세로, 가로, 채널)
    # 예: (16, 112, 112, 3)
    frames_np = np.array(frames)
    
    # 5. PyTorch 3D CNN 입력 형태인 (채널, 프레임수, 세로, 가로)로 차원 순서 변경
    # (16, 112, 112, 3) -> (3, 16, 112, 112)
    frames_np = np.transpose(frames_np, (3, 0, 1, 2))
    
    # 6. PyTorch Tensor로 변환하고 맨 앞에 배치(Batch) 차원 추가 
    # 최종 형태 -> (1, 3, 16, 112, 112)
    tensor_data = torch.from_numpy(frames_np).unsqueeze(0)
    
    return tensor_data

if __name__ == "__main__":
    # 코드가 잘 작동하는지 확인하기 위한 테스트 로직
    print("✅ 영상 전처리 모듈이 정상적으로 로드되었습니다!")
    
    # 실제 영상 파일이 아직 없으므로, OpenCV가 처리했다고 가정한 가짜 텐서를 출력해 봅니다.
    dummy_video_tensor = torch.rand(1, 3, 16, 112, 112)
    
    print("-" * 50)
    print(f"최종 변환된 텐서의 형태: {dummy_video_tensor.shape}")
    print("[해석] (배치크기 1, RGB채널 3, 연속프레임 16개, 세로 112, 가로 112)")
    print("-" * 50)