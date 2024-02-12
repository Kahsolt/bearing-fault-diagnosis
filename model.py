#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/01

from utils import *


# ref: https://github.com/xiaosongshine/bearing_detection_by_conv1d
class NaiveConv1d(nn.Module):

  def __init__(self, num_classes=10):
    super().__init__()

    # x1024 downsample: [1, 6000] => [512, 94]
    self.features = nn.Sequential(
      nn.Conv1d(  1,  16, kernel_size=8, stride=2),            nn.ReLU(inplace=True),
      nn.Conv1d( 16,  16, kernel_size=8, stride=2, padding=3), nn.ReLU(inplace=True), nn.MaxPool1d(kernel_size=2),
      nn.Conv1d( 16,  64, kernel_size=4, stride=2, padding=2), nn.ReLU(inplace=True),
      nn.Conv1d( 64,  64, kernel_size=4, stride=2, padding=2), nn.ReLU(inplace=True), nn.MaxPool1d(kernel_size=2),
      nn.Conv1d( 64, 256, kernel_size=4, stride=2, padding=2), nn.ReLU(inplace=True),
      nn.Conv1d(256, 256, kernel_size=4, stride=2, padding=2), nn.ReLU(inplace=True), nn.MaxPool1d(kernel_size=2),
      nn.Conv1d(256, 512, kernel_size=2, stride=1, padding=1), nn.ReLU(inplace=True),
      nn.Conv1d(512, 512, kernel_size=2, stride=1, padding=1), nn.ReLU(inplace=True), nn.MaxPool1d(kernel_size=2),
    )
    self.avgpool = nn.AdaptiveAvgPool1d(1)
    self.dropout = nn.Dropout(0.3)
    self.fc = nn.Linear(512, num_classes)

  def forward(self, x:Tensor) -> Tensor:
    x = self.features(x)  # [B, C=512, L=7]
    x = self.avgpool(x)   # [B, C=512, L=1]
    x = x.view(x.size(0), -1)
    x = self.dropout(x)   # [B, C=512]
    return self.fc(x)     # [B, NC]


class Naive4Conv1d(nn.Module):

  base_cls = NaiveConv1d

  def __init__(self, num_classes=4):
    super().__init__()

    self.base = NaiveConv1d()
    self.fc = nn.Sequential(
      nn.Linear(10, 16),
      nn.ReLU(),
      nn.Linear(16, num_classes),
    )

  def load_weights(self, state_dict:Dict[str, Tensor]):
    self.base.load_state_dict(state_dict)
    self.base.requires_grad_(False)

  def forward(self, x:Tensor) -> Tensor:
    x = self.base(x)  # [B, D=10]
    return self.fc(x) # [B, NC=4]


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

    # x64 downsample: [F=129, L=256] => [F=2, L=4]
    self.features = nn.Sequential(
      nn.Conv2d(  1,  16, kernel_size=7, stride=2, padding=3), nn.BatchNorm2d(16),  nn.SiLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2),
      nn.Conv2d( 16,  64, kernel_size=5, stride=2, padding=2), nn.BatchNorm2d(64),  nn.SiLU(inplace=True),
      nn.Conv2d( 64, 256, kernel_size=5, stride=2, padding=2), nn.BatchNorm2d(256), nn.SiLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2),
      nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(512), nn.SiLU(inplace=True),
    )
    self.avgpool = nn.AdaptiveMaxPool2d((1, 1))
    self.dropout = nn.Dropout(0.75)
    self.fc = nn.Linear(512, num_classes)

  def forward(self, x:Tensor) -> Tensor:
    x = self.features(x)  # [B, C=512, F=2, L=4]
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
  #model = SimpleConv1d()
  #X = torch.randn(1, 1, 4096)
  model = NaiveConv1d()
  X = torch.randn(1, 1, 6000)
  logits = model(X)
  print(model)
  print('X.shape:', X.shape)
  print('logits.shape:', logits.shape)
