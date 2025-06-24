import torch
from torch import nn
import torch.nn.functional as F


class MixPoolClassifier(nn.Module):
    """
    混合池化方法MixPool
    """
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.num_classes = num_classes
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        # x: (B, C, H, W)
        avg = F.adaptive_avg_pool2d(x, 1)  # (B, C, 1, 1)
        maxp = F.adaptive_max_pool2d(x, 1) # (B, C, 1, 1)
        feat = self.alpha * maxp + (1 - self.alpha) * avg
        feat = feat.view(x.size(0), -1)    # (B, C)
        return self.fc(feat)               # (B, num_classes)