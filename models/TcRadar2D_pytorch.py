import torch
from torch import nn, einsum
import numpy as np
from einops import rearrange, repeat



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
    indices = torch.tensor(np.array([[x, y] for x in range(window_size) for y in range(window_size)]),dtype=torch.long)
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

class Tc_Radar2D(nn.Module):
    def __init__(self, *, hidden_dim, layers, heads, channels=3, num_classes=1000, head_dim=32, window_size=7,
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
        self.patch_partition2 = PatchMerging(in_channels=hidden_dim, out_channels=hidden_dim*2,
                                            downscaling_factor=downscaling_factors[1])

        self.en3 = StageModule(in_channels=hidden_dim * 2, hidden_dimension=hidden_dim * 4, layers=layers[2],
                                  downscaling_factor=downscaling_factors[2], num_heads=heads[2], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        self.patch_partition3 = PatchMerging(in_channels=hidden_dim * 2, out_channels=hidden_dim*4,
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

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 8),
            nn.Linear(hidden_dim * 8, num_classes)
        )



    def forward(self, img):
        # Encoder start
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
        dec4 = self.de4(enc4)
        dec3 = self.de3(dec4)
        dec2 = self.de2(dec3)
        dec1 = self.de1(dec2)
        # Decoder end

        out = dec1.mean(dim=[2, 3])
        return self.mlp_head(out)


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
        num_classes=3,
        head_dim=32,
        window_size=7,
        downscaling_factors=(4, 2, 2, 2),
        relative_pos_embedding=True
    )
    dummy_x = torch.randn(2, 3, 224, 224)
    logits = net(dummy_x)  # (2,3)
    print(net)
    print(logits)
