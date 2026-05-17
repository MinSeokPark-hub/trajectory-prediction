from fastapi import FastAPI
from pydantic import BaseModel
import torch

# 방금 우리가 만든 추론 파이프라인을 불러옵니다!
from app.inference import predict_trajectory

app = FastAPI(title="Pedestrian Trajectory Prediction API")

# 외부에서 요청을 보낼 때 사용할 데이터 양식
class VideoRequest(BaseModel):
    video_path: str = "data/sample_video.mp4"

@app.get("/")
async def root():
    return {"message": "3D CNN 궤적 예측 API 서버가 정상적으로 실행되었습니다."}

@app.get("/health")
async def health_check():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    return {
        "status": "ok",
        "device_used": device,
        "message": "모델 추론 준비 완료"
    }

# 🚀 새로운 핵심 기능: 영상 경로를 주면 궤적을 예측해서 돌려주는 API
@app.post("/predict")
async def predict_endpoint(request: VideoRequest):
    # inference.py의 함수를 실행하고 그 결과를 그대로 반환합니다.
    result = predict_trajectory(request.video_path)
    return result