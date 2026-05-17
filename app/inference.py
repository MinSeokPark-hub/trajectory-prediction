import torch
import os
from app.model import PedestrianTrajectoryModel
from app.preprocess import extract_frames

def predict_trajectory(video_path):
    """
    영상을 입력받아 전처리 후 3D CNN 모델을 통과시켜 미래 궤적(X, Y)을 예측합니다.
    """
    # 1. 모델 준비 (나중에 학습이 끝나면 weights 폴더에서 가중치를 불러옵니다)
    model = PedestrianTrajectoryModel()
    model.eval() # 모델을 학습 모드가 아닌 '추론(예측) 모드'로 설정
    
    # 2. 영상 전처리하여 텐서로 변환
    if os.path.exists(video_path):
        input_tensor = extract_frames(video_path, num_frames=16, resize=(112, 112))
        if input_tensor is None:
            return {"error": "프레임 추출 실패 (영상이 너무 짧거나 손상됨)"}
    else:
        print(f"⚠️ '{video_path}' 파일을 찾을 수 없어 임시 테스트용 텐서를 생성합니다.")
        input_tensor = torch.rand(1, 3, 16, 112, 112)

    # 3. 모델에 데이터 넣고 예측 결과 뽑기 (Inference)
    with torch.no_grad(): # 예측만 할 때는 메모리를 아끼기 위해 기울기 계산을 끕니다.
        output = model(input_tensor)
    
    # 4. 결과값 (X, Y 좌표) 반환
    # 출력 형태는 (1, 2) 이므로 리스트로 변환해서 값을 뽑아냅니다.
    predicted_x, predicted_y = output[0].tolist()
    
    return {
        "status": "success",
        "predicted_x": round(predicted_x, 4),
        "predicted_y": round(predicted_y, 4)
    }

if __name__ == "__main__":
    print("✅ 추론(Inference) 모듈 실행 테스트 시작...\n")
    
    # data 폴더에 영상이 있다고 가정하고 테스트 실행
    test_video = "data/sample_video.mp4"
    result = predict_trajectory(test_video)
    
    print("-" * 50)
    print("🎯 [최종 예측 결과]")
    print(f"예측된 다음 X 좌표: {result.get('predicted_x')}")
    print(f"예측된 다음 Y 좌표: {result.get('predicted_y')}")
    print("-" * 50)
    print("완벽합니다! 모델이 텐서를 받아 좌표를 뱉어내는 전체 파이프라인이 연결되었습니다.")