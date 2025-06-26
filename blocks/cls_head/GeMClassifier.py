import torch
from torch import nn
import torch.nn.functional as F


class GeMClassifier(nn.Module):
    """
    混合池化方法Gem
    """
    def __init__(self, in_channels, num_classes, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        # x: (B, C, H, W)
        x = x.clamp(min=self.eps).pow(self.p)
        x = F.adaptive_avg_pool2d(x, 1)     # (B, C, 1, 1)
        feat = x.pow(1.0 / self.p).view(x.size(0), -1)  # (B, C)
        return self.fc(feat)                # (B, num_classes)