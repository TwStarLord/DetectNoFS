import torch
from torch import nn, einsum
import numpy as np
from einops import rearrange
import torch.nn.functional as F

# from blocks.DIFB.DIFB2D import DIFB2D, DIFB2D_v1
from blocks.ca.MutilHeadCrossAttentionBlock import MutilHeadCrossAttentionBlock
from blocks.cls_head.GeMClassifier import GeMClassifier
from blocks.cls_head.LiteTransformerPoolCls import LiteTransformerPoolCls
from blocks.cls_head.MixPoolClassifier import MixPoolClassifier
from blocks.cls_head.PatchTransformerPoolCls import PatchTransformerPoolCls
from blocks.cls_head.SAOLPoolCls import SAOLPoolCls
from blocks.cls_head.SPPoolCls import SPPoolCls
from blocks.cls_head.TransformerPoolCls import TransformerPoolCls
from blocks.feature_fusion.ConvFeatureFsuion import ConvFeatureFusion
from blocks.upsample.PatchExpandUpsample import PatchExpandUpsample


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


class Tc_Radar2D(nn.Module):
    def __init__(self, *, module_register, hidden_dim, layers, heads, channels=3, num_classes=5, head_dim=32,
                 window_size=7,
                 downscaling_factors=(4, 2, 2, 2), relative_pos_embedding=True):
        super().__init__()
        # 模块字典，动态构建
        self.mreg = module_register

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

        # self.patch3 = PatchExpandPixelShuffle(hidden_dim * 8, hidden_dim * 4, scale=2)
        # self.patch3 = PatchExpandConvTranspose(hidden_dim * 8, hidden_dim * 4, scale=2)

        self.ca3 = MutilHeadCrossAttentionBlock(hidden_dim * 4, hidden_dim * 4, embed_dim=hidden_dim * 4, num_heads=4)
        self.conv3 = nn.Conv2d(hidden_dim * 4, hidden_dim * 4, 3, padding=1)

        # self.patch2 = PatchExpandPixelShuffle(hidden_dim * 4, hidden_dim * 2, scale=2)
        # self.patch2 = PatchExpandConvTranspose(hidden_dim * 4, hidden_dim * 2, scale=2)

        self.ca2 = MutilHeadCrossAttentionBlock(hidden_dim * 2, hidden_dim * 2, embed_dim=hidden_dim * 2, num_heads=4)
        self.conv2 = nn.Conv2d(hidden_dim * 2, hidden_dim * 2, 3, padding=1)

        # self.patch1 = PatchExpandPixelShuffle(hidden_dim * 2, hidden_dim, scale=2)
        # self.patch1 = PatchExpandConvTranspose(hidden_dim * 2, hidden_dim, scale=2)

        self.conv1 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.final = nn.Conv2d(hidden_dim, num_classes, 1)

        # ==============================动态模块统一在此处调整 start==============================
        # self.difb1 = DIFB2D(hidden_dim)
        # self.difb2 = DIFB2D(hidden_dim * 2)
        # self.difb3 = DIFB2D(hidden_dim * 4)
        # self.difb4 = DIFB2D(hidden_dim * 8)

        # # DIFB-v1
        # self.difb1 = DIFB2D_v1(hidden_dim, hidden_dim // 4)
        # self.difb2 = DIFB2D_v1(hidden_dim * 2, hidden_dim // 2)
        # self.difb3 = DIFB2D_v1(hidden_dim * 4, hidden_dim)
        # self.difb4 = DIFB2D_v1(hidden_dim * 8, hidden_dim * 2)

        # 1. DIFB 模块，v1配置参考上述代码
        self.difb1 = self.mreg.build_from_config(
            category='DIFB',
            in_chans=hidden_dim
        )
        self.difb2 = self.mreg.build_from_config(
            category='DIFB',
            in_chans=hidden_dim * 2
        )
        self.difb3 = self.mreg.build_from_config(
            category='DIFB',
            in_chans=hidden_dim * 4
        )
        self.difb4 = self.mreg.build_from_config(
            category='DIFB',
            in_chans=hidden_dim * 8
        )

        # 2. 特征融合模块
        # self.fusion1 = AFFFeatureFusion(hidden_dim)
        # self.fusion2 = AFFFeatureFusion(hidden_dim * 2)
        # self.fusion3 = AFFFeatureFusion(hidden_dim * 4)
        # self.fusion4 = AFFFeatureFusion(hidden_dim * 8)

        # self.fusion1 = ConvFeatureFusion(hidden_dim)
        # self.fusion2 = ConvFeatureFusion(hidden_dim * 2)
        # self.fusion3 = ConvFeatureFusion(hidden_dim * 4)
        # self.fusion4 = ConvFeatureFusion(hidden_dim * 8)

        self.fusion1 = self.mreg.build_from_config(
            category='feature_fusion',
        )
        self.fusion2 = self.mreg.build_from_config(
            category='feature_fusion',
        )
        self.fusion3 = self.mreg.build_from_config(
            category='feature_fusion',
        )
        self.fusion4 = self.mreg.build_from_config(
            category='feature_fusion',
        )

        # 3. 交叉注意力模块
        # self.ca1 = MutilHeadCrossAttentionBlock(enc_channels=hidden_dim, dec_channels=hidden_dim, embed_dim=hidden_dim,
        #                                         num_heads=4)

        self.ca = self.mreg.build_from_config(
            category='ca',
            enc_channels=hidden_dim, dec_channels=hidden_dim, embed_dim=hidden_dim,
            num_heads=4
        )
        # 4. 上采样模块
        # self.patch3 = PatchExpandUpsample(in_channels= hidden_dim* 8, out_channels= hidden_dim* 4, scale=2)
        # self.patch2 = PatchExpandUpsample(hidden_dim * 4, hidden_dim * 2, scale=2)
        # self.patch1 = PatchExpandUpsample(hidden_dim * 2, hidden_dim, scale=2)

        self.patch3 = self.mreg.build_from_config(
            category='upsample',
            in_channels=hidden_dim * 8, out_channels=hidden_dim * 4, scale=2
        )
        self.patch2 = self.mreg.build_from_config(
            category='upsample',
            in_channels=hidden_dim * 4, out_channels=hidden_dim * 2, scale=2
        )
        self.patch1 = self.mreg.build_from_config(
            category='upsample',
            in_channels=hidden_dim * 2, out_channels=hidden_dim, scale=2
        )
        # 5. 分类头
        # # TODO 不同的聚合模块
        # # 空间注意力聚合模块
        # self.saolpoolcls = SAOLPoolCls(in_channels=hidden_dim, num_classes=num_classes)
        # # 空间金字塔池化(SPP, ASPP)
        # self.sppoolcls = SPPoolCls(in_channels=hidden_dim, num_classes=num_classes)
        # # 混合池化方法(MixPool、GeM等)
        # self.mixpoolcls = MixPoolClassifier(in_channels=hidden_dim, num_classes=num_classes)
        # self.gemcls = GeMClassifier(in_channels=hidden_dim, num_classes=num_classes)
        # # 类别Token / Transformer池化
        # self.transformerpoolcls = TransformerPoolCls(in_channels=hidden_dim, num_classes=num_classes)
        # self.litetransformerpoolcls = LiteTransformerPoolCls(in_channels=hidden_dim, num_classes=num_classes)
        # self.patchtransformerpoolcls = PatchTransformerPoolCls(in_channels=hidden_dim, num_classes=num_classes)

        self.cls_head = self.mreg.build_from_config(
            category='cls_head',
            in_channels=hidden_dim,
            num_classes=num_classes
        )
        # ==============================动态模块统一在此处调整 end==============================

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
        enc1 = self.fusion1(t1, d1)
        # stage1
        p2 = self.patch_partition2(enc1)
        t2 = self.en2(p2)
        d2 = self.difb2(p2.permute(0, 3, 1, 2))
        enc2 = self.fusion2(t2, d2)
        # stage1
        p3 = self.patch_partition3(enc2)
        t3 = self.en3(p3)
        d3 = self.difb3(p3.permute(0, 3, 1, 2))
        enc3 = self.fusion3(t3, d3)
        # stage1
        p4 = self.patch_partition4(enc3)
        t4 = self.en4(p4)
        d4 = self.difb4(p4.permute(0, 3, 1, 2))
        enc4 = self.fusion4(t4, d4)
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
        # out = self.saolpoolcls(de1)
        # out = self.sppoolcls(de1)
        # out = self.mixpoolcls(de1)
        # out = self.gemscls(de1)
        # out = self.litetransformerpoolcls(de1)
        out = self.patchtransformerpoolcls(de1)

        # out = self.transformerpool(de1)# RAM OOM

        return out


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
