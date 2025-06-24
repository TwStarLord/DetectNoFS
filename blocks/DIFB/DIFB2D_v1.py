import torch
from torch import nn
import torch.nn.functional as F

# TODO DIFB内部改进，此处可以结合一维卷积块、自适应多频率通道空间注意力模块实现多维信息捕捉和总结

class DIFB2D_v1(nn.Module):
    """
    该DIFB示例包含三个不同膨胀率分支，分支间信息交织后拼接降维，可直接用于增强特征。
通过以上设计，可灵活地将DIFB和交叉注意力结合使用。例如，在Transformer编码器中插入DIFB以获取多尺度卷积特征，再在解码交叉注意力中利用这些特征完成全局上下文融合，从而提高特征表达能力和分类性能
。上述PyTorch示例提供了可供参考的实现思路。
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(in_ch, out_ch, 3, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(in_ch, out_ch, 3, padding=3, dilation=3)
        self.conv_cat = nn.Conv2d(out_ch * 3, in_ch, 1)

    def forward(self, x):
        b1 = F.relu(self.conv1(x))
        b2 = F.relu(self.conv2(x) + b1)  # 支路间信息融合
        b3 = F.relu(self.conv3(x) + b2)
        fused = torch.cat([b1, b2, b3], dim=1)
        return self.conv_cat(fused)
