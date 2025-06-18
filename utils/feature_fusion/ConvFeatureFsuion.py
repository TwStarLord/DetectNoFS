import torch
from torch import nn


class ConvFeatureFusion(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch*2, ch, 1)
    def forward(self, feat_trans, feat_difb):
        fused = torch.cat([feat_trans, feat_difb], dim=1)
        fused = self.conv(fused)
        return fused