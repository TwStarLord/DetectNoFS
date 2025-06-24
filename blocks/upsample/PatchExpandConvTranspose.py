from torch import nn


class PatchExpandConvTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, scale=2):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=scale, stride=scale)

    def forward(self, x):
        # x: [B, C_in, H, W]
        x = self.deconv(x)  # [B, C_out, H*r, W*r]
        return x