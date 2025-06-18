import torch
from torch import nn


class TransformerPoolCls(nn.Module):
    """
    类别Token / Transformer池化
    """
    def __init__(self, in_channels, num_classes, num_layers=1, num_heads=8):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, in_channels))
        encoder_layer = nn.TransformerEncoderLayer(d_model=in_channels, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        # x: (B, C, H, W) -> 转为 (B, H*W, C)
        B, C, H, W = x.shape
        tokens = x.view(B, C, H * W).permute(0, 2, 1)  # (B, H*W, C)
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, C)
        tokens = torch.cat([cls, tokens], dim=1)  # (B, 1+H*W, C)
        # Transformer 输入需要 (seq_len, batch, dim)
        tokens = tokens.permute(1, 0, 2)  # (1+H*W, B, C)
        enc = self.transformer(tokens)  # (1+H*W, B, C)
        enc = enc.permute(1, 0, 2)  # (B, 1+H*W, C)
        cls_out = enc[:, 0]  # (B, C)
        return self.fc(cls_out)  # (B, num_classes)