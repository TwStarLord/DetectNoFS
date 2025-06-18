import torch
from torch import nn
import torch.nn.functional as F

# utils/difb/DIFB2D_v1.py
from utils.registry import register_module

@register_module
class DIFB2D(nn.Module):
    """Dense Information Fusion Block (2D 卷积版)"""

    def __init__(self, in_chans):
        super().__init__()
        C = in_chans
        # 每个分支输出通道 = C//4
        out_ch = C // 4
        # Branch1: 3×3, dilation=1, padding=1
        self.b1 = nn.Conv2d(C, out_ch, kernel_size=3,
                            stride=1, padding=1, dilation=1)

        # Branch2:
        # b2_1: 3×3, dilation=2, padding=2
        self.b2_1 = nn.Conv2d(C, out_ch, kernel_size=3,
                              stride=1, padding=2, dilation=2)
        # b2_2: 5×5, dilation=2, padding=4
        self.b2_2 = nn.Conv2d(C, out_ch, kernel_size=5,
                              stride=1, padding=4, dilation=2)

        # Branch3:
        # b3_1: 3×3, dilation=4, padding=4
        self.b3_1 = nn.Conv2d(C, out_ch, kernel_size=3,
                              stride=1, padding=4, dilation=4)
        # b3_2: 5×5, dilation=4, padding=8
        self.b3_2 = nn.Conv2d(C, out_ch, kernel_size=5,
                              stride=1, padding=8, dilation=4)

        # Branch4:
        # b4_1: 3×3, dilation=6, padding=6
        self.b4_1 = nn.Conv2d(C, out_ch, kernel_size=3,
                              stride=1, padding=6, dilation=6)
        # b4_2: 5×5, dilation=6, padding=12
        self.b4_2 = nn.Conv2d(C, out_ch, kernel_size=5,
                              stride=1, padding=12, dilation=6)

        # Fuse: 1×1 卷积将 7*(C/4) 通道融合回 C 通道
        self.fuse = nn.Conv2d(7 * out_ch, C,
                              kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """
        输入 x: Tensor, shape = (B, C, H, W)
        """
        b1 = self.b1(x)
        # b2_1 和 b2_2 输出 (B, C/4, H, W)，cat 后 b2 通道数 = C/2
        b2 = torch.cat([self.b2_1(x), self.b2_2(x)], dim=1)
        b3 = torch.cat([self.b3_1(x), self.b3_2(x)], dim=1)
        b4 = torch.cat([self.b4_1(x), self.b4_2(x)], dim=1)

        # fuse 输入通道数 = C/4 + C/2 + C/2 + C/2 = 7C/4
        out = self.fuse(torch.cat([b1, b2, b3, b4], dim=1))
        # 输出 shape = (B, C, H, W)
        return out
