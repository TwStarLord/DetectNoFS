import torch
import torch.nn as nn
from models.swin_transformer import PatchEmbed, PatchMerging, BasicLayer


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
            nn.BatchNorm3d(C // 4),
            nn.ReLU(True)
        )
        # Branch2
        self.b2_1 = nn.Sequential(
            nn.Conv3d(C, C // 4, 3, padding=(1, 2, 2), dilation=(1, 2, 2)),
            nn.BatchNorm3d(C // 4),
            nn.ReLU(True)
        )
        self.b2_2 = nn.Sequential(
            nn.Conv3d(C, C // 4, (3, 5, 5), padding=(2, 4, 4), dilation=(2, 2, 2)),
            nn.BatchNorm3d(C // 4),
            nn.ReLU(True)
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
            nn.BatchNorm3d(C // 4),
            nn.ReLU(True)
        )
        self.b4_2 = nn.Sequential(
            nn.Conv3d(C, C // 4, (3, 5, 5), padding=(2, 12, 12), dilation=(2, 6, 6)),
            nn.BatchNorm3d(C // 4),
            nn.ReLU(True)
        )
        self.fuse = nn.Conv3d(C, C, 1)  # Eq.(5)

    def forward(self, x):
        b1 = self.b1(x)
        b2 = torch.cat([self.b2_1(x), self.b2_2(x)], dim=1)
        b3 = torch.cat([self.b3_1(x), self.b3_2(x)], dim=1)
        b4 = torch.cat([self.b4_1(x), self.b4_2(x)], dim=1)
        return self.fuse(torch.cat([b1, b2, b3, b4], dim=1))


# ================= CrossAttention3D =================
# Ref: Section 3.3, Eq.(6)-(11)
class CrossAttention3D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)

        self.conv3D = nn.Conv3d(kernel_size=1)


    def forward(self, x_dec, x_enc):
        x_cat = torch.cat([x_dec, x_enc], dim=1)
        x_conv = self.conv3D(x_cat)



        Q = self.q(x_dec)  # Eq.(10)
        K = self.k(x_enc)  # Eq.(9)
        V = self.v(x_enc)
        attn = (Q @ K.transpose(-2, -1)) / (self.dim ** 0.5)  # Eq.(7)
        attn = attn.softmax(dim=-1)
        x = attn @ V
        return self.out(x) + x_dec  # Eq.(11)


# =================== TC–Radar Model ===================
class TC_Radar(nn.Module):
    """
    Full TC–Radar Encoder-Decoder Network
    Ref: Section 3.1, Figure 2, Section 3.2, Section 3.3
    Input: (B, C, D, H, W)
    Output: (B, num_classes, D, H, W)
    """

    def __init__(self, in_chans=2, embed_dim=32, depths=[2, 2, 6],
                 num_heads=[4, 8, 16], window_size=8,
                 img_size=128, patch_size=4, num_classes=3):
        super().__init__()
        self.embed_dim = embed_dim
        # ---- Encoder ----
        # Stage1 PatchEmbed
        self.patch_embed1 = PatchEmbed(
            img_size=img_size, patch_size=patch_size,
            in_chans=in_chans, embed_dim=embed_dim
        )  # Sec3.1.1
        self.norm1 = nn.LayerNorm(embed_dim)
        self.layer1 = BasicLayer(
            dim=embed_dim, input_resolution=(img_size // patch_size, img_size // patch_size),
            depth=depths[0], num_heads=num_heads[0], window_size=window_size, downsample=None
        )
        self.difb1 = DIFB3D(embed_dim)
        self.alpha1 = nn.Parameter(torch.tensor(0.5))
        # Stage2 PatchMerging
        self.patch_merge2 = PatchMerging(
            input_resolution=(img_size // patch_size, img_size // patch_size), dim=embed_dim
        )
        self.norm2 = nn.LayerNorm(embed_dim * 2)
        self.layer2 = BasicLayer(
            dim=embed_dim * 2, input_resolution=(img_size // (patch_size * 2), img_size // (patch_size * 2)),
            depth=depths[1], num_heads=num_heads[1], window_size=window_size, downsample=None
        )
        self.difb2 = DIFB3D(embed_dim * 2)
        self.alpha2 = nn.Parameter(torch.tensor(0.5))
        # Stage3 PatchMerging
        self.patch_merge3 = PatchMerging(
            input_resolution=(img_size // (patch_size * 2), img_size // (patch_size * 2)), dim=embed_dim * 2
        )
        self.norm3 = nn.LayerNorm(embed_dim * 4)
        self.layer3 = BasicLayer(
            dim=embed_dim * 4, input_resolution=(img_size // (patch_size * 4), img_size // (patch_size * 4)),
            depth=depths[2], num_heads=num_heads[2], window_size=window_size, downsample=None
        )
        self.difb3 = DIFB3D(embed_dim * 4)
        self.alpha3 = nn.Parameter(torch.tensor(0.5))
        # ---- Decoder ----
        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.ca2 = CrossAttention3D(embed_dim * 2)
        self.lin2 = nn.Conv3d(embed_dim * 2, embed_dim * 2, kernel_size=1)
        self.up1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.ca1 = CrossAttention3D(embed_dim)
        self.lin1 = nn.Conv3d(embed_dim, embed_dim, kernel_size=1)
        self.up0 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.classifier = nn.Conv3d(embed_dim, num_classes, kernel_size=1)  # Sec3.1.2

    def forward(self, x):
        B, C, D, H, W = x.shape
        # --- Stage1 ---
        x1 = x.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)  # (B*D, C, H, W)
        t1 = self.patch_embed1(x1)
        t1 = self.norm1(t1)
        t1 = self.layer1(t1)  # (B*D, N1, E)
        fm1 = t1.permute(0, 2, 1).reshape(B, D, -1, H // 2, W // 2).permute(0, 2, 1, 3, 4)
        c1 = self.difb1(fm1)
        enc1 = self.alpha1 * c1 + (1 - self.alpha1) * fm1  # (B, E, D/2, H/2, W/2)
        # --- Stage2 ---
        t2 = self.patch_merge2(t1);
        t2 = self.norm2(t2)
        t2 = self.layer2(t2)
        fm2 = t2.permute(0, 2, 1).reshape(B, D, -1, H // 4, W // 4).permute(0, 2, 1, 3, 4)
        c2 = self.difb2(fm2)
        enc2 = self.alpha2 * c2 + (1 - self.alpha2) * fm2  # (B,2E,D/4,H/4,W/4)
        # --- Stage3 ---
        t3 = self.patch_merge3(t2);
        t3 = self.norm3(t3)
        t3 = self.layer3(t3)
        fm3 = t3.permute(0, 2, 1).reshape(B, D, -1, H // 8, W // 8).permute(0, 2, 1, 3, 4)
        c3 = self.difb3(fm3)
        enc3 = self.alpha3 * c3 + (1 - self.alpha3) * fm3  # (B,4E,D/8,H/8,W/8)
        # --- Decoder Stage2 ---
        u2 = self.up2(enc3)  # -> (B,4E,D/4,H/4,W/4)
        U2 = u2.view(B, 4 * self.embed_dim, -1).permute(0, 2, 1)
        E2 = enc2.view(B, 2 * self.embed_dim, -1).permute(0, 2, 1)
        ca2 = self.ca2(U2, E2).permute(0, 2, 1).view(B, 2 * self.embed_dim, D, H // 4, W // 4)
        d2 = self.lin2(ca2)
        # --- Decoder Stage1 ---
        u1 = self.up1(d2)  # -> (B,2E,D/2,H/2,W/2)
        U1 = u1.view(B, 2 * self.embed_dim, -1).permute(0, 2, 1)
        E1 = enc1.view(B, self.embed_dim, -1).permute(0, 2, 1)
        ca1 = self.ca1(U1, E1).permute(0, 2, 1).view(B, self.embed_dim, D, H // 2, W // 2)
        d1 = self.lin1(ca1)
        # --- Final Upsample & Classify ---
        u0 = self.up0(d1)  # -> (B, E, D, H, W)
        out = self.classifier(u0)  # (B,num_classes,D,H,W)
        return out


if __name__ == '__main__':
    model = TC_Radar(in_chans=2)
    x = torch.randn(1, 2, 16, 128, 128)
    y = model(x)
    print(y.shape)  # expect (1,3,16,128,128)
