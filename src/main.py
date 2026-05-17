from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from src.inference_engine import InferenceEngine
import numpy as np

app = FastAPI(title="Safe-AI Trajectory API")
engine = InferenceEngine()

class PedestrianData(BaseModel):
    # 과거 5프레임의 좌표 데이터
    history_3d: List[List[float]] 
    history_pix: List[List[float]]

@app.post("/predict")
async def predict_ttc(data: PedestrianData):
    try:
        # 리스트를 넘파이 배열로 변환하여 추론
        ttc, status = engine.predict(
            np.array(data.history_pix), 
            np.array(data.history_3d)
        )
        return {"ttc": round(ttc, 2), "status": status}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 서버 실행 (터미널): uvicorn src.main:app --host 0.0.0.0 --port 8888