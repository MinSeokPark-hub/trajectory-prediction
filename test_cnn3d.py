import torch
from src.predictors.cnn3d_predictor import CNN3DPredictor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = CNN3DPredictor(in_channels=3, T=4, H=64, W=64).to(device)

x = torch.randn(2, 3, 4, 64, 64).to(device)

# 13~14번: Forward 통과 및 차원 검증
out = model(x)
assert out.shape == (2, 1), f"출력 shape 불일치: {out.shape}"
print(f"✅ Forward 통과: 입력 {x.shape} → 출력 {out.shape}")

# 16~17번: 어텐션 인터페이스 전달 테스트
feat = model.extract_features(x)
print(f"✅ 특징 맵 shape: {feat.shape}")
assert feat.shape[0] == 2,   "배치 크기 불일치"
assert feat.shape[1] == 128, "채널 수 불일치"
assert torch.isfinite(feat).all(), "특징 맵에 NaN/Inf 포함"
print("✅ 어텐션 인터페이스 데이터 누락 없이 전달 완료")