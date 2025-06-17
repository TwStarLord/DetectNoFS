import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, einsum

from models.TcRadar3D import CrossAttention3D


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

class CyclicShift(nn.Module):
    def __init__(self, displacement):
        super().__init__()
        self.displacement = displacement

    def forward(self, x):
        return torch.roll(x, shifts=(self.displacement, self.displacement), dims=(1, 2))

def get_relative_distances(window_size):
    indices = torch.tensor(np.array([[x, y] for x in range(window_size) for y in range(window_size)]),dtype=torch.long)
    distances = indices[None, :, :] - indices[:, None, :]
    return distances

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

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

        self.patch_partition = PatchMerging(in_channels=in_channels, out_channels=hidden_dimension,
                                            downscaling_factor=downscaling_factor)

        self.layers = nn.ModuleList([])
        for _ in range(layers // 2):
            self.layers.append(nn.ModuleList([
                SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=False, window_size=window_size, relative_pos_embedding=relative_pos_embedding),
                SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=True, window_size=window_size, relative_pos_embedding=relative_pos_embedding),
            ]))

    def forward(self, x):
        x = self.patch_partition(x)
        for regular_block, shifted_block in self.layers:
            x = regular_block(x)
            x = shifted_block(x)
        return x.permute(0, 3, 1, 2)

# ==================== DIFB3D Implementation ====================
# Ref: Section 3.2, Eq.(2)-(5), Table 1
class DIFB3D(nn.Module):
    """Dense Information Fusion Block"""

    def __init__(self, in_chans):
        super().__init__()
        C = in_chans
        # Branch1
        self.b1 = nn.Sequential(
            nn.Conv3d(C, C // 4, 3, padding=1, dilation=1),
            nn.BatchNorm3d(C // 4), nn.ReLU(True)
        )
        # Branch2
        self.b2_1 = nn.Sequential(
            nn.Conv3d(C, C // 4, 3, padding=(1, 2, 2), dilation=(1, 2, 2)),
            nn.BatchNorm3d(C // 4), nn.ReLU(True)
        )
        self.b2_2 = nn.Sequential(
            nn.Conv3d(C, C // 4, (3, 5, 5), padding=(2, 4, 4), dilation=(2, 2, 2)),
            nn.BatchNorm3d(C // 4), nn.ReLU(True)
        )
        # Branch3
        self.b3_1 = nn.Sequential(
            nn.Conv3d(C, C // 4, 3, padding=(1, 4, 4), dilation=(1, 4, 4)),
            nn.BatchNorm3d(C // 4), nn.ReLU(True)
        )
        self.b3_2 = nn.Sequential(
            nn.Conv3d(C, C // 4, (3, 5, 5), padding=(2, 8, 8), dilation=(2, 4, 4)),
            nn.BatchNorm3d(C // 4), nn.ReLU(True)
        )
        # Branch4
        self.b4_1 = nn.Sequential(
            nn.Conv3d(C, C // 4, 3, padding=(1, 6, 6), dilation=(1, 6, 6)),
            nn.BatchNorm3d(C // 4), nn.ReLU(True)
        )
        self.b4_2 = nn.Sequential(
            nn.Conv3d(C, C // 4, (3, 5, 5), padding=(2, 12, 12), dilation=(2, 6, 6)),
            nn.BatchNorm3d(C // 4), nn.ReLU(True)
        )
        self.fuse = nn.Conv3d(7 * C // 4, C, 1)  # Eq.(5)

    def forward(self, x):
        b1 = self.b1(x)
        b2 = torch.cat([self.b2_1(x), self.b2_2(x)], dim=1)
        b3 = torch.cat([self.b3_1(x), self.b3_2(x)], dim=1)
        b4 = torch.cat([self.b4_1(x), self.b4_2(x)], dim=1)
        return self.fuse(torch.cat([b1, b2, b3, b4], dim=1))


# =================== TC–Radar with SwinTransformer ===================
class TC_Radar(nn.Module):
    """Transformer–CNN Hybrid Network based on Swin architecture"""

    def __init__(self, in_chans=3, hidden_dim=96, layers=(2, 2, 6, 2), heads=(3, 6, 12, 24),
                 head_dim=32, window_size=7, downscale=(4, 2, 2, 2), upscale=[2, 2, 2], num_classes=5):
        super().__init__()
        # StageModules from swin_transformer_pytorch
        self.trm1 = StageModule(
            in_channels=in_chans,
            hidden_dimension=hidden_dim,
            layers=layers[0],
            downscaling_factor=downscale[0],
            num_heads=heads[0],
            head_dim=head_dim,
            window_size=window_size,
            relative_pos_embedding=True
        )
        self.trm2 = StageModule(
            in_channels=hidden_dim,
            hidden_dimension=hidden_dim * 2,
            layers=layers[1],
            downscaling_factor=downscale[1],
            num_heads=heads[1],
            head_dim=head_dim,
            window_size=window_size,
            relative_pos_embedding=True
        )
        self.trm3 = StageModule(
            in_channels=hidden_dim * 2,
            hidden_dimension=hidden_dim * 4,
            layers=layers[2],
            downscaling_factor=downscale[2],
            num_heads=heads[2],
            head_dim=head_dim,
            window_size=window_size,
            relative_pos_embedding=True
        )
        # DIFB and fusion
        self.difb1 = DIFB3D(hidden_dim)
        self.difb2 = DIFB3D(hidden_dim * 2)
        self.difb3 = DIFB3D(hidden_dim * 4)

        self.lamuda1 = nn.Parameter(torch.tensor(0.5))
        self.lamuda2 = nn.Parameter(torch.tensor(0.5))
        self.lamuda3 = nn.Parameter(torch.tensor(0.5))

        self.alpha1 = nn.Parameter(torch.tensor(0.5))
        self.alpha2 = nn.Parameter(torch.tensor(0.5))
        self.alpha3 = nn.Parameter(torch.tensor(0.5))
        # Decoder similar to paper (upsample + cross-attn)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=(1, upscale[2], upscale[2]), mode='trilinear', align_corners=False),
            nn.Conv3d(in_channels=hidden_dim * 4, out_channels=hidden_dim * 2, kernel_size=1)
        )
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=(1, upscale[1], upscale[1]), mode='trilinear', align_corners=False),
            nn.Conv3d(in_channels=hidden_dim * 2, out_channels=hidden_dim, kernel_size=1)
        )
        self.up0 = nn.Sequential(
            nn.Upsample(scale_factor=(1, upscale[2], upscale[2]), mode='trilinear', align_corners=False),
            # nn.Conv3d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1)
        )

        self.ca2 = CrossAttention3D(dim=hidden_dim * 2)
        self.ca1 = CrossAttention3D(dim=hidden_dim)
        self.lin2 = nn.Conv3d(hidden_dim * 4, hidden_dim * 2, 1)
        self.lin1 = nn.Conv3d(hidden_dim * 2, hidden_dim, 1)
        self.classifier = nn.Conv3d(hidden_dim, num_classes, 1)

    def forward(self, x):
        # x: (B, C, D,H, W)
        B, C, D, H, W = x.shape
        # Encoder start
        # Stage1 transformer+DIFB input size：(B, C,D,H, W)；output size：(B, C,D/2,H/2, W/2)

        e1 = self.trm1(x.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W))
        d1 = self.difb1(e1.reshape(B, e1.shape[1], -1, e1.shape[2], e1.shape[3])).permute(0, 2, 1, 3, 4).reshape(B * D,
                                                                                                                 e1.shape[
                                                                                                                     1],
                                                                                                                 e1.shape[
                                                                                                                     2],
                                                                                                                 e1.shape[
                                                                                                                     3])
        enc1 = self.lamuda1 * d1 + (1 - self.lamuda1) * e1
        # Stage2 transformer+DIFB input size：(B, C,D,H, W)；output size：(B, 2C,D/4,H/4, W/4)
        e2 = self.trm2(enc1)
        d2 = self.difb2(e2.reshape(B, e2.shape[1], -1, e2.shape[2], e2.shape[3])).permute(0, 2, 1, 3, 4).reshape(B * D,
                                                                                                                 e2.shape[
                                                                                                                     1],
                                                                                                                 e2.shape[
                                                                                                                     2],
                                                                                                                 e2.shape[
                                                                                                                     3])
        enc2 = self.lamuda2 * d2 + (1 - self.lamuda2) * e2
        # Stage3 transformer+DIFB input size：(B, C,D,H, W)；output size：(B, 4C,D/4,H/8, W/8)
        e3 = self.trm3(enc2)
        d3 = self.difb3(e3.reshape(B, e3.shape[1], -1, e3.shape[2], e3.shape[3])).permute(0, 2, 1, 3, 4).reshape(B * D,
                                                                                                                 e3.shape[
                                                                                                                     1],
                                                                                                                 e3.shape[
                                                                                                                     2],
                                                                                                                 e3.shape[
                                                                                                                     3])
        enc3 = self.lamuda3 * d3 + (1 - self.lamuda3) * e3
        # Encoder end

        # Decoder
        # decoder接收的输入包括两部分：1编码器对应层的输出2上一层解码器的输出
        # u2 = self.up2(enc3.reshape(B, -1,enc3.shape[1], enc3.shape[2], enc3.shape[3]).permute(0, 2, 1, 3, 4))
        u2 = self.up2(enc3.reshape(B, -1, enc3.shape[1], enc3.shape[2], enc3.shape[3]).permute(0, 2, 1, 3, 4))
        # dec2 = self.lin2(u2.reshape(B, u2.shape[1], -1, u2.shape[2], u2.shape[3]).permute(0, 2, 1, 3, 4))
        # dec2 = self.lin2(u2.reshape(B, u2.shape[1], -1, u2.shape[2], u2.shape[3]).permute(0, 2, 1, 3, 4))
        ca2 = self.ca2(u2.transpose(1, 2).flatten(2),
                       enc2.reshape(B, D, -1, enc2.shape[1], enc2.shape[2], enc2.shape[3]).flatten(2))
        # dec2 = self.lin2(ca2.transpose(1, 2).view_as(enc2))

        u1 = self.up1(ca2)
        dec1 = self.lin1(u1.transpose(1, 2).view_as(enc1))
        ca1 = self.ca1(u1.flatten(2).transpose(1, 2), enc1.flatten(2).transpose(1, 2))
        # dec1 = self.lin1(ca1.transpose(1, 2).view_as(enc1))

        u0 = self.up0(dec1.unsqueeze(2)).squeeze(2)
        out = self.classifier(u0.unsqueeze(2)).squeeze(2)
        return out


if __name__ == '__main__':
    model = TC_Radar()
    img = torch.randn(2, 3, 4, 224, 224)
    y = model(img)
    print(y.shape)  # (2, num_classes, 224,224)

    # net = SwinTransformer(
    #     hidden_dim=96,
    #     layers=(2, 2, 6, 2),
    #     heads=(3, 6, 12, 24),
    #     channels=3,
    #     num_classes=3,
    #     head_dim=32,
    #     window_size=7,
    #     downscaling_factors=(4, 2, 2, 2),
    #     relative_pos_embedding=True
    # )
    # dummy_x = torch.randn(1, 3, 224, 224)
    # logits = net(dummy_x)  # (1,3)
    # print(net)
    # print(logits)
