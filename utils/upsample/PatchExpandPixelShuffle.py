from torch import nn


class PatchExpandPixelShuffle(nn.Module):
    def __init__(self, in_channels, out_channels, scale=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * scale * scale, kernel_size=1)
        self.shuffle = nn.PixelShuffle(scale)

    def forward(self, x):
        # x: [B, C_in, H, W]
        x = self.conv(x)  # [B, C_out*r^2, H, W]
        x = self.shuffle(x)  # [B, C_out, H*r, W*r]
        return x