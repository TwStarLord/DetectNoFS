import torch
from torch import nn


class LiteTransformerPoolCls(nn.Module):
    """
    轻量化类别Token / Transformer池化
    先对特征图做自适应平均池化降到 (r x r)，再扁平化加上 cls token，
    用标准 TransformerEncoder 进行注意力聚合，最后线性映射得到分类 logits。
    """
    def __init__(self, in_channels, num_classes, pool_size=8, num_heads=8, num_layers=1):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        self.cls_token = nn.Parameter(torch.randn(1, 1, in_channels))
        encoder_layer = nn.TransformerEncoderLayer(d_model=in_channels, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, _, _ = x.shape
        # 1) 降采样到 pool_size x pool_size
        x = self.pool(x)  # (B, C, r, r)
        # 2) 扁平化空间维度
        x = x.flatten(2).permute(0, 2, 1)  # (B, r*r, C)
        # 3) 拼接 cls token
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, C)
        x = torch.cat([cls, x], dim=1)          # (B, 1 + r*r, C)
        # 4) Transformer（batch_first=True）
        x = self.transformer(x)                 # (B, 1 + r*r, C)
        # 5) 取 cls 位置
        cls_out = x[:, 0]                       # (B, C)
        # 6) 分类 head
        logits = self.fc(cls_out)               # (B, num_classes)
        return logits