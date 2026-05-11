# 이미지 및 GPS 데이터 기반의 이동체 경로예측 및 충돌예측 연구

09조 · 조우연 · 박민석 · 허민경 · 지도교수: 이규철 교수님

---

## 프로젝트 개요

KITTI Tracking / ETH·UCY 데이터셋을 기반으로 보행자·차량의 미래 궤적을 예측하고,
TTC(Time-to-Collision)를 산출하여 3단계 위험도(Safe / Warning / Danger)로 분류하는 시스템.

---

## 시스템 구조

입력(KITTI 이미지 + 좌표) → Preprocessing(KittiParser / SGANParser / BaseParser) → 3개 모델 병렬 추론 → InferenceEngine 앙상블 → FastAPI / QueuePipeline 출력

---

## 모델 구조

### 1. SimpleTTC (FC Network)
- 입력: BBox 5값 (x, y, w, h, depth) × 3프레임 = 15차원
- 구조: Linear(15→60) → ReLU × 8층 → Linear(60→1)
- 출력: TTC (초)
- 학습: KITTI, MSELoss, Adam

### 2. LSTM Trajectory Predictor
- 입력: (B, 5, 3) — 과거 5프레임의 (x, y, depth)
- 구조: LSTM(input=3, hidden=128, layers=2) → Linear(128→15)
- 출력: 미래 5프레임 좌표 → ΔD/ΔV 방식으로 TTC 산출
- 학습: KITTI, MSELoss, Adam

### 3. CNN3DPredictor (Sprint 1 신규)
- 입력: (B, 3, T, 64, 64) — RGB 이미지 T프레임 시퀀스
- 구조
  - Conv3d(3→32) → BN → ReLU → MaxPool3d(1,2,2)
  - Conv3d(32→64) → BN → ReLU → MaxPool3d(2,2,2)
  - Conv3d(64→128) → BN → ReLU
  - SpatioTemporalAttention(128) — 채널 어텐션
  - AdaptiveAvgPool3d(1,1,1)
  - Linear(128→64) → ReLU → Dropout(0.3) → Linear(64→1)
- 출력: TTC (초)
- 어텐션 특징 맵 shape: (B, 128, T', H', W')

### SpatioTemporalAttention
- GAP → Linear(C→C//4) → ReLU → Linear(C//4→C) → Sigmoid
- 채널별 중요도 가중치를 학습하여 특징 맵에 적용

### InferenceEngine 앙상블
- 세 모델 TTC 중 min값 선택 (가장 위험한 값 우선)
- Danger: TTC ≤ 1.5s / Warning: TTC ≤ 3.0s / Safe: TTC > 3.0s

---

## 데이터셋

| 데이터셋 | 형식 | 용도 |
|---------|------|------|
| KITTI Tracking | 이미지 + 라벨 (.txt) | SimpleTTC / LSTM / 3D CNN 학습 |
| ETH/UCY (SGAN) | 궤적 좌표 (.txt) | 보행자 궤적 검증 |

---

## 디렉토리 구조

- src/predictors/ — simple_ttc.py / lstm_predictor.py / cnn3d_predictor.py
- src/utils/ — base_parser.py / kitti_parser.py / sgan_parser.py / data_helper.py / dataset.py / lstm_dataset.py / cnn3d_dataset.py
- src/pipeline/ — queue_pipeline.py
- src/inference_engine.py — 앙상블 추론 엔진
- src/main.py — FastAPI 서버

---

## 실행 방법

환경 설정
```bash
conda activate pedestrian
```

FastAPI 서버 실행
```bash
cd /workspace/minseok_park
uvicorn src.main:app --host 0.0.0.0 --port 8888 --reload
```

3D CNN 학습
```bash
python src/train_cnn3d.py
```

동작 확인
```bash
python test_run.py      # 파서 테스트
python test_cnn3d.py    # 3D CNN Forward 검증
python test_queue.py    # Queue 파이프라인 검증
```

---

## Sprint 1 완료 항목

| 항목 | 상태 |
|------|------|
| KITTI / ETH·UCY 데이터 로더 | 완료 |
| BaseParser 통합 인터페이스 | 완료 |
| 프레임 시퀀스 추출 및 텐서 변환 | 완료 |
| 3D CNN 아키텍처 + 채널 어텐션 | 완료 |
| Forward 차원 검증 | 완료 |
| FastAPI 비동기 서버 | 완료 |
| FIFO Queue 스트리밍 파이프라인 | 완료 |
| GPU 환경 확인 (Tesla P100 × 8) | 완료 |