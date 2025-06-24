from torch import nn


class SAOLPoolCls(nn.Module):
    """
    Spatial Attention and Output Layer Pooling
    """
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv_att = nn.Conv2d(in_channels, 1, kernel_size=1)  # 生成空间注意力图
        self.conv_cls = nn.Conv2d(in_channels, num_classes, kernel_size=1)  # 生成每位置的类别预测

    def forward(self, x):
        # x: (B, C, H, W)
        A = self.conv_att(x)  # (B, 1, H, W)
        # 对空间位置做 softmax 归一化
        A = A.view(x.size(0), -1).softmax(dim=-1).view(x.size(0), 1, x.size(2), x.size(3))
        Y = self.conv_cls(x)  # (B, num_classes, H, W)
        # 按空间位置加权求和
        out = (A * Y).view(x.size(0), Y.size(1), -1).sum(dim=-1)  # (B, num_classes)
        return out