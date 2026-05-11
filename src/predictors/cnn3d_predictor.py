import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatioTemporalAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.fc  = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, channels // 4),
            nn.ReLU(),
            nn.Linear(channels // 4, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(self.gap(x))
        w = w.view(w.size(0), w.size(1), 1, 1, 1)
        return x * w


class CNN3DPredictor(nn.Module):
    def __init__(self, in_channels=3, T=4, H=64, W=64):
        super().__init__()

        self.conv3d_1 = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
        )
        self.conv3d_2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
        )
        self.conv3d_3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(),
        )

        self.attention = SpatioTemporalAttention(channels=128)

        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def extract_features(self, x):
        x = self.conv3d_1(x)
        x = self.conv3d_2(x)
        x = self.conv3d_3(x)
        x = self.attention(x)
        return x

    def forward(self, x):
        x = self.extract_features(x)
        x = self.pool(x)
        return self.regressor(x)
