import torch
from torch import nn


class AFFFeatureFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = torch.sigmoid(nn.Parameter(torch.zeros(1)))
    def forward(self, feat_trans, feat_difb):
        fused = self.alpha * feat_trans + (1-self.alpha) * feat_difb
        return fused