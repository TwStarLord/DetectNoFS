import torch
from torch import nn, einsum
import numpy as np
from einops import rearrange, repeat
import torch.nn.functional as F


class CyclicShift(nn.Module):
    def __init__(self, displacement):
        super().__init__()
        self.displacement = displacement

    def forward(self, x):
        return torch.roll(x, shifts=(self.displacement, self.displacement), dims=(1, 2))


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


def create_mask(window_size, displacement, upper_lower, left_right):
    mask = torch.zeros(window_size ** 2, window_size ** 2)

    if upper_lower:
        mask[-displacement * window_size:, :-displacement * window_size] = float('-inf')
        mask[:-displacement * window_size, -displacement * window_size:] = float('-inf')

    if left_right:
        mask = rearrange(mask, '(h1 w1) (h2 w2) -> h1 w1 h2 w2', h1=window_size, h2=window_size)
        mask[:, -displacement:, :, :-displacement] = float('-inf')
        mask[:, :-displacement, :, -displacement:] = float('-inf')
        mask = rearrange(mask, 'h1 w1 h2 w2 -> (h1 w1) (h2 w2)')

    return mask


def get_relative_distances(window_size):
    indices = torch.tensor(np.array([[x, y] for x in range(window_size) for y in range(window_size)]), dtype=torch.long)
    distances = indices[None, :, :] - indices[:, None, :]
    return distances


class WindowAttention(nn.Module):
    def __init__(self, dim, heads, head_dim, shifted, window_size, relative_pos_embedding):
        super().__init__()
        inner_dim = head_dim * heads

        self.heads = heads
        self.scale = head_dim ** -0.5
        self.window_size = window_size
        self.relative_pos_embedding = relative_pos_embedding
        self.shifted = shifted

        if self.shifted:
            displacement = window_size // 2
            self.cyclic_shift = CyclicShift(-displacement)
            self.cyclic_back_shift = CyclicShift(displacement)
            self.upper_lower_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement,
                                                             upper_lower=True, left_right=False), requires_grad=False)
            self.left_right_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement,
                                                            upper_lower=False, left_right=True), requires_grad=False)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        if self.relative_pos_embedding:
            self.relative_indices = get_relative_distances(window_size) + window_size - 1
            self.pos_embedding = nn.Parameter(torch.randn(2 * window_size - 1, 2 * window_size - 1))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(window_size ** 2, window_size ** 2))

        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        if self.shifted:
            x = self.cyclic_shift(x)

        b, n_h, n_w, _, h = *x.shape, self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        nw_h = n_h // self.window_size
        nw_w = n_w // self.window_size

        q, k, v = map(
            lambda t: rearrange(t, 'b (nw_h w_h) (nw_w w_w) (h d) -> b h (nw_h nw_w) (w_h w_w) d',
                                h=h, w_h=self.window_size, w_w=self.window_size), qkv)

        dots = einsum('b h w i d, b h w j d -> b h w i j', q, k) * self.scale

        if self.relative_pos_embedding:
            dots += self.pos_embedding[self.relative_indices[:, :, 0], self.relative_indices[:, :, 1]]
        else:
            dots += self.pos_embedding

        if self.shifted:
            dots[:, :, -nw_w:] += self.upper_lower_mask
            dots[:, :, nw_w - 1::nw_w] += self.left_right_mask

        attn = dots.softmax(dim=-1)

        out = einsum('b h w i j, b h w j d -> b h w i d', attn, v)
        out = rearrange(out, 'b h (nw_h nw_w) (w_h w_w) d -> b (nw_h w_h) (nw_w w_w) (h d)',
                        h=h, w_h=self.window_size, w_w=self.window_size, nw_h=nw_h, nw_w=nw_w)
        out = self.to_out(out)

        if self.shifted:
            out = self.cyclic_back_shift(out)
        return out


class SwinBlock(nn.Module):
    def __init__(self, dim, heads, head_dim, mlp_dim, shifted, window_size, relative_pos_embedding):
        super().__init__()
        self.attention_block = Residual(PreNorm(dim, WindowAttention(dim=dim,
                                                                     heads=heads,
                                                                     head_dim=head_dim,
                                                                     shifted=shifted,
                                                                     window_size=window_size,
                                                                     relative_pos_embedding=relative_pos_embedding)))
        self.mlp_block = Residual(PreNorm(dim, FeedForward(dim=dim, hidden_dim=mlp_dim)))

    def forward(self, x):
        x = self.attention_block(x)
        x = self.mlp_block(x)
        return x


class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels, downscaling_factor):
        super().__init__()
        self.downscaling_factor = downscaling_factor
        self.patch_merge = nn.Unfold(kernel_size=downscaling_factor, stride=downscaling_factor, padding=0)
        self.linear = nn.Linear(in_channels * downscaling_factor ** 2, out_channels)

    def forward(self, x):
        b, c, h, w = x.shape
        new_h, new_w = h // self.downscaling_factor, w // self.downscaling_factor
        x = self.patch_merge(x).view(b, -1, new_h, new_w).permute(0, 2, 3, 1)
        x = self.linear(x)
        return x


class StageModule(nn.Module):
    def __init__(self, in_channels, hidden_dimension, layers, downscaling_factor, num_heads, head_dim, window_size,
                 relative_pos_embedding):
        super().__init__()
        assert layers % 2 == 0, 'Stage layers need to be divisible by 2 for regular and shifted block.'

        # self.patch_partition = PatchMerging(in_channels=in_channels, out_channels=hidden_dimension,
        #                                     downscaling_factor=downscaling_factor)

        self.layers = nn.ModuleList([])
        for _ in range(layers // 2):
            self.layers.append(nn.ModuleList([
                SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=False, window_size=window_size, relative_pos_embedding=relative_pos_embedding),
                SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=True, window_size=window_size, relative_pos_embedding=relative_pos_embedding),
            ]))

    def forward(self, x):
        # x = self.patch_partition(x)
        for regular_block, shifted_block in self.layers:
            x = regular_block(x)
            x = shifted_block(x)
        return x.permute(0, 3, 1, 2)


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


# TODO 以下三种上采样方式可以进行对比
"""
Decoder Patch Expand 1
"""


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


"""
Decoder Patch Expand 2
"""


class PatchExpandConvTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, scale=2):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=scale, stride=scale)

    def forward(self, x):
        # x: [B, C_in, H, W]
        x = self.deconv(x)  # [B, C_out, H*r, W*r]
        return x


"""
Decoder Patch Expand 3
"""


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


# TODO 后期可以尝试其他注意力或者其他多个融合注意力
class MutilHeadCrossAttentionBlock(nn.Module):
    """
    多头注意力实现交叉注意力
    """

    def __init__(self, enc_channels, dec_channels, embed_dim, num_heads):
        super().__init__()
        self.q_proj = nn.Linear(dec_channels, embed_dim)
        self.k_proj = nn.Linear(enc_channels, embed_dim)
        self.v_proj = nn.Linear(enc_channels, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.out_proj = nn.Linear(embed_dim, dec_channels)

    def forward(self, enc_feat, dec_feat):
        # enc_feat: [B, C_e, H, W], dec_feat: [B, C_d, H, W]
        B, C_e, H, W = enc_feat.shape
        _, C_d, _, _ = dec_feat.shape
        # 展平空间维度为序列： (H*W, B, C)
        enc_flat = enc_feat.flatten(2).permute(2, 0, 1)  # [HW, B, C_e]
        dec_flat = dec_feat.flatten(2).permute(2, 0, 1)  # [HW, B, C_d]
        # 线性映射到相同的嵌入维度
        Q = self.q_proj(dec_flat)  # [HW, B, E]
        K = self.k_proj(enc_flat)
        V = self.v_proj(enc_flat)
        # 跨注意力：查询来自解码器，键值来自编码器
        attn_out, _ = self.attn(Q, K, V)
        # 恢复形状并通道映射回解码器通道数
        attn_out = self.out_proj(attn_out)  # [HW, B, C_d]
        attn_out = attn_out.permute(1, 2, 0).view(B, C_d, H, W)
        # 残差融合
        return attn_out + dec_feat


class Tc_Radar2D(nn.Module):
    def __init__(self, *, hidden_dim, layers, heads, channels=3, num_classes=5, head_dim=32, window_size=7,
                 downscaling_factors=(4, 2, 2, 2), relative_pos_embedding=True):
        super().__init__()

        self.en1 = StageModule(in_channels=channels, hidden_dimension=hidden_dim, layers=layers[0],
                               downscaling_factor=downscaling_factors[0], num_heads=heads[0], head_dim=head_dim,
                               window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        self.patch_partition1 = PatchMerging(in_channels=channels, out_channels=hidden_dim,
                                             downscaling_factor=downscaling_factors[0])

        self.en2 = StageModule(in_channels=hidden_dim, hidden_dimension=hidden_dim * 2, layers=layers[1],
                               downscaling_factor=downscaling_factors[1], num_heads=heads[1], head_dim=head_dim,
                               window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        self.patch_partition2 = PatchMerging(in_channels=hidden_dim, out_channels=hidden_dim * 2,
                                             downscaling_factor=downscaling_factors[1])

        self.en3 = StageModule(in_channels=hidden_dim * 2, hidden_dimension=hidden_dim * 4, layers=layers[2],
                               downscaling_factor=downscaling_factors[2], num_heads=heads[2], head_dim=head_dim,
                               window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        self.patch_partition3 = PatchMerging(in_channels=hidden_dim * 2, out_channels=hidden_dim * 4,
                                             downscaling_factor=downscaling_factors[2])

        self.en4 = StageModule(in_channels=hidden_dim * 4, hidden_dimension=hidden_dim * 8, layers=layers[3],
                               downscaling_factor=downscaling_factors[3], num_heads=heads[3], head_dim=head_dim,
                               window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        self.patch_partition4 = PatchMerging(in_channels=hidden_dim * 4, out_channels=hidden_dim * 8,
                                             downscaling_factor=downscaling_factors[3])

        self.difb1 = DIFB2D(hidden_dim)
        self.difb2 = DIFB2D(hidden_dim * 2)
        self.difb3 = DIFB2D(hidden_dim * 4)
        self.difb4 = DIFB2D(hidden_dim * 8)

        self.lamuda1 = nn.Parameter(torch.tensor(0.5))
        self.lamuda2 = nn.Parameter(torch.tensor(0.5))
        self.lamuda3 = nn.Parameter(torch.tensor(0.5))
        self.lamuda4 = nn.Parameter(torch.tensor(0.5))

        # self.mlp_head = nn.Sequential(
        #     nn.LayerNorm(hidden_dim * 8),
        #     nn.Linear(hidden_dim * 8, num_classes)
        # )

        self.patch3 = PatchExpandPixelShuffle(hidden_dim * 8, hidden_dim * 4, scale=2)  # 或 ConvTranspose/Upsample
        self.ca3 = MutilHeadCrossAttentionBlock(hidden_dim * 4, hidden_dim * 4, embed_dim=hidden_dim * 4, num_heads=4)
        self.conv3 = nn.Conv2d(hidden_dim * 4, hidden_dim * 4, 3, padding=1)

        self.patch2 = PatchExpandPixelShuffle(hidden_dim * 4, hidden_dim * 2, scale=2)
        self.ca2 = MutilHeadCrossAttentionBlock(hidden_dim * 2, hidden_dim * 2, embed_dim=hidden_dim * 2, num_heads=4)
        self.conv2 = nn.Conv2d(hidden_dim * 2, hidden_dim * 2, 3, padding=1)

        self.patch1 = PatchExpandPixelShuffle(hidden_dim * 2, hidden_dim, scale=2)  # 若有 enc0 则继续，否则视为输出直至输入分辨率
        self.ca1 = MutilHeadCrossAttentionBlock(hidden_dim, hidden_dim, embed_dim=hidden_dim, num_heads=4)
        self.conv1 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.final = nn.Conv2d(hidden_dim, num_classes, 1)

        # TODO 不同的聚合模块
        # 空间注意力聚合模块
        self.saolpool = SAOLPool(in_channels=hidden_dim, num_classes=num_classes)
        # 空间金字塔池化(SPP, ASPP)
        self.sppool = SPPool(in_channels=hidden_dim, num_classes=num_classes)
        # 混合池化方法(MixPool、GeM等)
        self.mixpool = MixPoolClassifier(in_channels=hidden_dim, num_classes=num_classes)
        self.gem = GeMClassifier(in_channels=hidden_dim, num_classes=num_classes)
        # 类别Token / Transformer池化
        self.transformerpool = TransformerPool(in_channels=hidden_dim, num_classes=num_classes)
        self.litetransformerpool = LiteTransformerPool(in_channels=hidden_dim, num_classes=num_classes)
        self.patchtransformerpool = PatchTransformerPool(in_channels=hidden_dim, num_classes=num_classes)

    def forward(self, img):
        # Encoder start
        # TODO 下述代码中每个stage中都是并行执行，后续可以换成串行执行，并进行对比，代码如下
        # t1 = self.en1(img)  # 已在内部完成了 downscaling
        # d1 = self.difb1(t1)
        # enc1 = ...

        # stage1
        p1 = self.patch_partition1(img)
        t1 = self.en1(p1)
        d1 = self.difb1(p1.permute(0, 3, 1, 2))
        enc1 = self.lamuda1 * d1 + (1 - self.lamuda1) * t1
        # stage1
        p2 = self.patch_partition2(enc1)
        t2 = self.en2(p2)
        d2 = self.difb2(p2.permute(0, 3, 1, 2))
        enc2 = self.lamuda2 * d2 + (1 - self.lamuda2) * t2
        # stage1
        p3 = self.patch_partition3(enc2)
        t3 = self.en3(p3)
        d3 = self.difb3(p3.permute(0, 3, 1, 2))
        enc3 = self.lamuda3 * d3 + (1 - self.lamuda3) * t3
        # stage1
        p4 = self.patch_partition4(enc3)
        t4 = self.en4(p4)
        d4 = self.difb4(p4.permute(0, 3, 1, 2))
        enc4 = self.lamuda4 * d4 + (1 - self.lamuda4) * t4
        # Encoder end

        # Decoder start
        pe3 = self.patch3(enc4)  # [B, C_d2, H2, W2]
        de3 = F.relu(self.conv3(self.ca3(enc3, pe3)))  # 融合并非线性变换
        pe2 = self.patch2(de3)  # [B, C_d1, H1, W1]
        de2 = F.relu(self.conv2(self.ca2(enc2, pe2)))
        pe1 = self.patch1(de2)
        de1 = F.relu(self.conv1(self.ca1(enc1, pe1)))
        # Decoder end

        # out = self.final(de1)  # [B, N_classes, H, W]
        # 此方法不推荐
        # out = out.mean(dim=[2, 3])

        # 全局最大池化(Global Max Pooling, GMP)

        # out = F.adaptive_avg_pool2d(out, 1).squeeze(-1).squeeze(-1)  # [B, C]

        # # 空间注意力聚合模块
        # self.saolpool = SAOLPool(in_channels=hidden_dim, num_classes=num_classes)
        # # 空间金字塔池化(SPP, ASPP)
        # self.sppool = SPPool(in_channels=hidden_dim, out_features=num_classes)
        # # 混合池化方法(MixPool、GeM等)
        # self.mixpool = MixPool()
        # self.gem = GeM()
        # # 类别Token / Transformer池化
        # self.transformerpool = TransformerPool(in_channels=hidden_dim, num_classes=num_classes)

        # TODO 对比
        # out = self.saolpool(de1)
        # out = self.sppool(de1)
        # out = self.mixpool(de1)
        # out = self.gem(de1)
        # out = self.litetransformerpool(de1)
        out = self.patchtransformerpool(de1)

        # out = self.transformerpool(de1)# RAM OOM


        return out


class SAOLPool(nn.Module):
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


class SPPool(nn.Module):
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


# 1) 定义 MixPool + FC
class MixPoolClassifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.num_classes = num_classes
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        # x: (B, C, H, W)
        avg = F.adaptive_avg_pool2d(x, 1)  # (B, C, 1, 1)
        maxp = F.adaptive_max_pool2d(x, 1) # (B, C, 1, 1)
        feat = self.alpha * maxp + (1 - self.alpha) * avg
        feat = feat.view(x.size(0), -1)    # (B, C)
        return self.fc(feat)               # (B, num_classes)

# 2) 定义 GeM + FC
class GeMClassifier(nn.Module):
    def __init__(self, in_channels, num_classes, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        # x: (B, C, H, W)
        x = x.clamp(min=self.eps).pow(self.p)
        x = F.adaptive_avg_pool2d(x, 1)     # (B, C, 1, 1)
        feat = x.pow(1.0 / self.p).view(x.size(0), -1)  # (B, C)
        return self.fc(feat)                # (B, num_classes)


class TransformerPool(nn.Module):
    """

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

class LiteTransformerPool(nn.Module):
    """
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

class PatchTransformerPool(nn.Module):
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



def swin_t(hidden_dim=96, layers=(2, 2, 6, 2), heads=(3, 6, 12, 24), **kwargs):
    return Tc_Radar2D(hidden_dim=hidden_dim, layers=layers, heads=heads, **kwargs)


def swin_s(hidden_dim=96, layers=(2, 2, 18, 2), heads=(3, 6, 12, 24), **kwargs):
    return Tc_Radar2D(hidden_dim=hidden_dim, layers=layers, heads=heads, **kwargs)


def swin_b(hidden_dim=128, layers=(2, 2, 18, 2), heads=(4, 8, 16, 32), **kwargs):
    return Tc_Radar2D(hidden_dim=hidden_dim, layers=layers, heads=heads, **kwargs)


def swin_l(hidden_dim=192, layers=(2, 2, 18, 2), heads=(6, 12, 24, 48), **kwargs):
    return Tc_Radar2D(hidden_dim=hidden_dim, layers=layers, heads=heads, **kwargs)


if __name__ == '__main__':
    net = Tc_Radar2D(
        hidden_dim=96,
        layers=(2, 2, 6, 2),
        heads=(3, 6, 12, 24),
        channels=3,
        num_classes=5,
        head_dim=32,
        window_size=7,
        downscaling_factors=(4, 2, 2, 2),
        relative_pos_embedding=True
    )
    dummy_x = torch.randn(10, 3, 224, 224)
    logits = net(dummy_x)  # (2,3)
    print(net)
    print(logits)
