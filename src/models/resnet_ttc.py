# src/models/resnet_ttc.py
"""
논문 3.3절 — 4가지 모델 변형
  A) image_only   : ResNet-18 이미지만 (512→256→MLP)
  B) dummy        : ego 특징 = 0 (20-dim zeros)
  C) velocity_only: ego 특징 = speed만 (나머지 0)
  D) full         : 전체 ego 20-dim

모든 변형이 동일 아키텍처를 공유, 입력만 다름.
"""
import torch
import torch.nn as nn
from torchvision import models


class EgoMotionEncoder(nn.Module):
    def __init__(self, input_dim: int = 20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)  # (B, 64)


class ResNetTTC(nn.Module):
    """
    Parameters
    ----------
    mode : 'image_only' | 'dummy' | 'velocity_only' | 'full'
    n_frames : 4 (논문 기본값)
    """
    def __init__(self, mode: str = 'full', n_frames: int = 4):
        super().__init__()
        assert mode in ('image_only', 'dummy', 'velocity_only', 'full')
        self.mode = mode
        self.n_frames = n_frames

        # ResNet-18 백본 (ImageNet pretrained)
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.image_encoder = nn.Sequential(*list(backbone.children())[:-1])  # → (B,512,1,1)
        self.image_proj = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        # Ego-motion MLP
        if mode != 'image_only':
            self.ego_encoder = EgoMotionEncoder(input_dim=20)
            fusion_dim = 256 + 64   # 320
        else:
            self.ego_encoder = None
            fusion_dim = 256

        # 최종 MLP
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Softplus(),  # TTC > 0 보장
        )

    def _prepare_ego(self, ego_raw: torch.Tensor) -> torch.Tensor:
        """
        ego_raw : (B, 20) — 원본 ego 20-dim 벡터
        mode에 따라 마스킹
        """
        if self.mode == 'dummy':
            return torch.zeros_like(ego_raw)
        if self.mode == 'velocity_only':
            # speed만 살림 (인덱스 0, 5, 10, 15 — 각 프레임의 첫 번째 특징)
            mask = torch.zeros_like(ego_raw)
            for i in range(4):
                mask[:, i * 5] = 1.0
            return ego_raw * mask
        return ego_raw  # 'full'

    def forward(self, images: torch.Tensor, ego: torch.Tensor) -> torch.Tensor:
        """
        images : (B, n_frames, 3, H, W)
        ego    : (B, 20)
        Returns: (B,) TTC 예측값
        """
        B, T, C, H, W = images.shape

        # 프레임별 이미지 인코딩 후 평균
        imgs = images.view(B * T, C, H, W)
        feats = self.image_encoder(imgs)          # (B*T, 512, 1, 1)
        feats = feats.view(B, T, 512).mean(dim=1) # (B, 512)
        img_feat = self.image_proj(feats)         # (B, 256)

        if self.mode == 'image_only':
            combined = img_feat
        else:
            ego_in = self._prepare_ego(ego)
            ego_feat = self.ego_encoder(ego_in)   # (B, 64)
            combined = torch.cat([img_feat, ego_feat], dim=1)  # (B, 320)

        out = self.head(combined).squeeze(-1)     # (B,)
        return out


def build_model(mode: str = 'full') -> ResNetTTC:
    return ResNetTTC(mode=mode)
