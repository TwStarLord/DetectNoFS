import torch
from torch import nn
import torch.nn.functional as F

class SPPoolCls(nn.Module):
    """
    空间金字塔池化(SPP, ASPP)
    """
    def __init__(self, in_channels, num_classes, pool_sizes=[1, 2, 4]):
        super().__init__()
        self.pool_sizes = pool_sizes
        self.fc = nn.Linear(in_channels * sum([s * s for s in pool_sizes]), num_classes)

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        outs = []
        for size in self.pool_sizes:
            pool = F.adaptive_avg_pool2d(x, (size, size))  # (B, C, size, size)
            outs.append(pool.view(B, -1))  # 展平
        feats = torch.cat(outs, dim=1)  # (B, C*(∑size^2))
        return self.fc(feats)  # (B, out_features), 视为 logits