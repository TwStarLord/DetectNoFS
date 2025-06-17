import torch
import torch.nn as nn
from TcRadar3D import DIFB3D

class TC_Radar_Encoder(nn.Module):
    """
    三层编码器：
      输入  (B, C, D,  H,  W)
    → Stage1 (B, E, D/2, H/2, W/2)
    → Stage2 (B,2E, D/4, H/4, W/4)
    → Stage3 (B,4E, D/8, H/8, W/8)
    并行Transformer & DIFB，加权融合。
    """
    def __init__(self,
                 in_chans=3,
                 embed_dim=96,
                 depths=(2,2,6),
                 num_heads=(3,6,12),
                 window_size=7,
                 img_size=128,
                 patch_size=4):
        super().__init__()

        # --- Stage1: PatchEmbed + BasicLayer (no downsample) ---
        self.patch_embed1 = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )  # 输出 tokens: (B·D, N1, E)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.layer1 = BasicLayer(
            dim=embed_dim,
            input_resolution=(img_size//patch_size, img_size//patch_size),
            depth=depths[0],
            num_heads=num_heads[0],
            window_size=window_size,
            downsample=None
        )
        self.difb1 = DIFB3D(embed_dim)
        self.alpha1 = nn.Parameter(torch.tensor(0.5))

        # --- Stage2: PatchMerging + BasicLayer ---
        self.patch_merge2 = PatchMerging(
            input_resolution=(img_size//patch_size, img_size//patch_size),
            dim=embed_dim
        )  # 空间降分辨率×2，通道维度2×E
        self.norm2 = nn.LayerNorm(embed_dim*2)
        self.layer2 = BasicLayer(
            dim=embed_dim*2,
            input_resolution=(img_size//(patch_size*2), img_size//(patch_size*2)),
            depth=depths[1],
            num_heads=num_heads[1],
            window_size=window_size,
            downsample=None
        )
        self.difb2 = DIFB3D(embed_dim*2)
        self.alpha2 = nn.Parameter(torch.tensor(0.5))

        # --- Stage3: PatchMerging + BasicLayer ---
        self.patch_merge3 = PatchMerging(
            input_resolution=(img_size//(patch_size*2), img_size//(patch_size*2)),
            dim=embed_dim*2
        )
        self.norm3 = nn.LayerNorm(embed_dim*4)
        self.layer3 = BasicLayer(
            dim=embed_dim*4,
            input_resolution=(img_size//(patch_size*4), img_size//(patch_size*4)),
            depth=depths[2],
            num_heads=num_heads[2],
            window_size=window_size,
            downsample=None
        )
        self.difb3 = DIFB3D(embed_dim*4)
        self.alpha3 = nn.Parameter(torch.tensor(0.5))


    def forward(self, x):
        # x: (B, C, D,  H,  W)
        B, C, D, H, W = x.shape

        # ---- Stage1 ----
        # (B, C, D, H, W) → (B·D, C, H, W)
        x1 = x.permute(0,2,1,3,4).reshape(B*D, C, H, W)

        # PatchEmbed → tokens1: (B·D, N1, E)
        tokens1 = self.patch_embed1(x1)
        tokens1 = self.norm1(tokens1)

        # Swin BasicLayer → out1: (B·D, N1, E)
        out1 = self.layer1(tokens1)

        # reshape tokens → feature map
        # N1 = (H/patch_size)*(W/patch_size)
        H1, W1 = H//2, W//2
        fm1 = out1.permute(0,2,1).reshape(B, D, -1, H1, W1).permute(0,2,1,3,4)
        # fm1: (B, E, D/2, H/2, W/2)

        # DIFB3D 卷积分支
        c1 = self.difb1(fm1)
        # 加权融合
        enc1 = self.alpha1 * c1 + (1-self.alpha1) * fm1
        # enc1: (B,   E, D/2,  H/2,  W/2)


        # ---- Stage2 ----
        # PatchMerging on tokens1 → t2: (B·D, N2, 2E)
        t2 = self.patch_merge2(out1)
        t2 = self.norm2(t2)
        # BasicLayer → out2: (B·D, N2, 2E)
        out2 = self.layer2(t2)

        # reshape → fm2: (B, 2E, D/4, H/4, W/4)
        H2, W2 = H//4, W//4
        fm2 = out2.permute(0,2,1).reshape(B, D, -1, H2, W2).permute(0,2,1,3,4)

        c2  = self.difb2(fm2)
        enc2 = self.alpha2 * c2 + (1-self.alpha2) * fm2
        # enc2: (B, 2E, D/4, H/4, W/4)


        # ---- Stage3 ----
        t3 = self.patch_merge3(out2)
        t3 = self.norm3(t3)
        out3 = self.layer3(t3)

        # reshape → fm3: (B, 4E, D/8, H/8, W/8)
        H3, W3 = H//8, W//8
        fm3 = out3.permute(0,2,1).reshape(B, D, -1, H3, W3).permute(0,2,1,3,4)

        c3  = self.difb3(fm3)
        enc3 = self.alpha3 * c3 + (1-self.alpha3) * fm3
        # enc3: (B, 4E, D/8, H/8, W/8)

        # 返回三层编码结果，供 Decoder 使用
        return enc1, enc2, enc3
