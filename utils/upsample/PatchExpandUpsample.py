from torch import nn


class PatchExpandUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale=2, mode='bilinear'):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale, mode=mode, align_corners=False)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # x: [B, C_in, H, W]
        x = self.upsample(x)  # [B, C_in, H*r, W*r]
        x = self.conv(x)  # [B, C_out, H*r, W*r]
        return x