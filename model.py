#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/01

from utils import *


class SimpleConv1d(nn.Module):

  def __init__(self, num_classes=4):
    super().__init__()

    # x1024 downsample: [1, 4096] => [512, 4]
    self.features = nn.Sequential(
      nn.Conv1d(  1,  16, kernel_size=16, stride=4, padding=8), nn.BatchNorm1d(16),  nn.SiLU(inplace=True), nn.MaxPool1d(kernel_size=2, stride=2),
      nn.Conv1d( 16,  64, kernel_size=8,  stride=4, padding=4), nn.BatchNorm1d(64),  nn.SiLU(inplace=True), nn.MaxPool1d(kernel_size=2, stride=2),
      nn.Conv1d( 64, 256, kernel_size=5,  stride=2, padding=2), nn.BatchNorm1d(256), nn.SiLU(inplace=True), nn.MaxPool1d(kernel_size=2, stride=2),
      nn.Conv1d(256, 512, kernel_size=3,  stride=2, padding=1), nn.BatchNorm1d(512), nn.SiLU(inplace=True), nn.MaxPool1d(kernel_size=2, stride=2),
    )
    self.avgpool = nn.AdaptiveAvgPool1d(1)
    self.dropout = nn.Dropout(0.75)
    self.fc = nn.Linear(512, num_classes)

  def forward(self, x:Tensor) -> Tensor:
    x = self.features(x)  # [B, C=512, L=4]
    x = self.avgpool(x)   # [B, C=512, L=1]
    x = x.view(x.size(0), -1)
    x = self.dropout(x)   # [B, C=512]
    return self.fc(x)     # [B, NC]


class SimpleConv2d(nn.Module):

  def __init__(self, num_classes=4):
    super().__init__()

    # x64 downsample: [F=257, L=128] => [F=4, L=6]
    self.features = nn.Sequential(
      nn.Conv2d(  1,  16, kernel_size=7, stride=2, padding=4), nn.BatchNorm2d(16),  nn.SiLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2),
      nn.Conv2d( 16,  64, kernel_size=5, stride=2, padding=3), nn.BatchNorm2d(64),  nn.SiLU(inplace=True),
      nn.Conv2d( 64, 256, kernel_size=5, stride=2, padding=3), nn.BatchNorm2d(256), nn.SiLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2),
      nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=2), nn.BatchNorm2d(512), nn.SiLU(inplace=True),
    )
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.dropout = nn.Dropout(0.75)
    self.fc = nn.Linear(512, num_classes)

  def forward(self, x:Tensor) -> Tensor:
    x = self.features(x)  # [B, C=512, F=4, L=6]
    x = self.avgpool(x)   # [B, C=512, F=1, L=1]
    x = x.view(x.size(0), -1)
    x = self.dropout(x)   # [B, C=512]
    return self.fc(x)     # [B, NC]


class MLP3d(nn.Module):

  def __init__(self, num_classes=4):
    super().__init__()

    self.mlp = nn.Sequential(
      nn.Linear(3, 128),
      nn.Dropout(0.5),
      nn.SiLU(inplace=True),
      nn.Linear(128, 128),
      nn.Dropout(0.5),
      nn.SiLU(inplace=True),
      nn.Linear(128, num_classes)
    )

  def forward(self, x:Tensor) -> Tensor:
    return self.mlp(x)


if __name__ == '__main__':
  model = SimpleConv1d()
  X = torch.randn(1, 1, 4096)
  logits = model(X)
  print(model)
  print('X.shape:', X.shape)
  print('logits.shape:', logits.shape)
