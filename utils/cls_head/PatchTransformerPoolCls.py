import torch
from torch import nn


class PatchTransformerPoolCls(nn.Module):
    """
    用卷积做补丁嵌入，步幅等于补丁大小；再加入 cls token 进入 Transformer 聚合。
    """
    def __init__(self, in_channels, num_classes, patch_size=8, num_heads=8, num_layers=1):
        super().__init__()
        self.patch_size = patch_size
        # 补丁映射：Conv2d(in, embed_dim, kernel=patch_size, stride=patch_size)
        # 这里我们直接保持 embed_dim = in_channels
        self.patch_embed = nn.Conv2d(in_channels, in_channels,
                                     kernel_size=patch_size,
                                     stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, in_channels))
        encoder_layer = nn.TransformerEncoderLayer(d_model=in_channels, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        B, C, H, W = x.shape
        # 1) Patch Embedding
        x = self.patch_embed(x)           # (B, C, H/p, W/p)
        h, w = x.shape[2], x.shape[3]
        x = x.flatten(2).permute(0,2,1)   # (B, (H/p)*(W/p), C)
        # 2) cls token
        cls = self.cls_token.expand(B, -1, -1)  # (B,1,C)
        x = torch.cat([cls, x], dim=1)          # (B,1+(H/p)*(W/p),C)
        # 3) Transformer
        x = self.transformer(x)                 # (B,1+...,C)
        cls_out = x[:,0]                        # (B, C)
        # 4) 分类 head
        return self.fc(cls_out)                # (B, num_classes)