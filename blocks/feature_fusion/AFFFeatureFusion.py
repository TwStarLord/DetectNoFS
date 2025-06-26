import torch
from torch import nn


class AFFFeatureFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha_raw = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def forward(self, feat_trans: torch.Tensor, feat_difb: torch.Tensor) -> torch.Tensor:
        # 在前向中对 raw 参数做 sigmoid，保证 alpha∈(0,1)
        alpha = torch.sigmoid(self.alpha_raw)
        # 此时 alpha、feat_trans、feat_difb 均在同一设备
        fused = alpha * feat_trans + (1 - alpha) * feat_difb
        return fused