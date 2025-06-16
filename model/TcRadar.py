import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================================
# 1. Dense Information Fusion Block (DIFB)
# Reference: Section 3.2, Eq.(2)-(5), Table 1 fileciteturn0file0
# ==========================================================
class DIFB3D(nn.Module):
    def __init__(self, in_channels):
        super(DIFB3D, self).__init__()
        C = in_channels
        # Branch1: dilation=(1,1,1)
        self.branch1 = nn.Sequential(
            nn.Conv3d(C, C // 4, kernel_size=3, padding=1, dilation=1),  # Table1 Branch1 fileciteturn0file0
            nn.BatchNorm3d(C // 4),
            nn.ReLU(inplace=True)
        )
        # Branch2: two paths (3x3x3,d=2) & (3x5x5,d=(2,2,2))
        self.branch2_1 = nn.Sequential(
            nn.Conv3d(C, C // 4, kernel_size=3, padding=(1, 2, 2), dilation=(1, 2, 2)),
            # Table1 Branch2-1 fileciteturn0file0
            nn.BatchNorm3d(C // 4),
            nn.ReLU(inplace=True)
        )
        self.branch2_2 = nn.Sequential(
            nn.Conv3d(C, C // 4, kernel_size=(3, 5, 5), padding=(2, 4, 4), dilation=(2, 4, 4)),
            # Table1 Branch2-2 fileciteturn0file0
            nn.BatchNorm3d(C // 4),
            nn.ReLU(inplace=True)
        )
        # Branch3 and Branch4 similar structure with larger dilation rates
        self.branch3_1 = nn.Sequential(
            nn.Conv3d(C, C // 4, kernel_size=3, padding=(1, 4, 4), dilation=(1, 4, 4)),
            # Table1 Branch3-1 fileciteturn0file0
            nn.BatchNorm3d(C // 4),
            nn.ReLU(inplace=True)
        )
        self.branch3_2 = nn.Sequential(
            nn.Conv3d(C, C // 4, kernel_size=(3, 5, 5), padding=(2, 8, 8), dilation=(2, 8, 8)),
            # Table1 Branch3-2 fileciteturn0file0
            nn.BatchNorm3d(C // 4),
            nn.ReLU(inplace=True)
        )
        self.branch4_1 = nn.Sequential(
            nn.Conv3d(C, C // 4, kernel_size=3, padding=(1, 6, 6), dilation=(1, 6, 6)),
            # Table1 Branch4-1 fileciteturn0file0
            nn.BatchNorm3d(C // 4),
            nn.ReLU(inplace=True)
        )
        self.branch4_2 = nn.Sequential(
            nn.Conv3d(C, C // 4, kernel_size=(3, 5, 5), padding=(2, 12, 12), dilation=(2, 12, 12)),
            # Table1 Branch4-2 fileciteturn0file0
            nn.BatchNorm3d(C // 4),
            nn.ReLU(inplace=True)
        )
        # 1x1x1 conv to fuse
        self.conv1x1 = nn.Conv3d(C, C, kernel_size=1)  # Eq.(5) fileciteturn0file0

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = torch.cat([self.branch2_1(x), self.branch2_2(x)], dim=1)
        b3 = torch.cat([self.branch3_1(x), self.branch3_2(x)], dim=1)
        b4 = torch.cat([self.branch4_1(x), self.branch4_2(x)], dim=1)
        out = torch.cat([b1, b2, b3, b4], dim=1)  # Eq.(5) fileciteturn0file0
        return self.conv1x1(out)


# ==========================================================
# 2. Cross-Attention (CA) Module
# Reference: Section 3.3, Eq.(6)-(11) fileciteturn0file0
# ==========================================================
class CrossAttention3D(nn.Module):
    def __init__(self, dim, num_heads=8, window_size=4):
        super(CrossAttention3D, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        # Linear projections for Q,K,V
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        # Output projection
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x_decoder, x_encoder):
        # x_decoder, x_encoder: (B, N, dim)
        Qd = self.q_proj(x_decoder)  # Eq.(10) fileciteturn0file0
        Ke = self.k_proj(x_encoder)  # Eq.(9) fileciteturn0file0
        Ve = self.v_proj(x_encoder)
        # Scaled dot-product attention
        attn = (Qd @ Ke.transpose(-2, -1)) / (self.dim ** 0.5)  # SA(Q,K,V) Eq.(7) fileciteturn0file0
        attn = attn.softmax(dim=-1)
        x_attn = attn @ Ve  # Eq.(9) fileciteturn0file0
        # Fuse with decoder residual (Eq.11)
        out = self.out_proj(x_attn) + x_decoder  # Eq.(11) fileciteturn0file0
        return out


# ==========================================================
# 3. TC–Radar Model Skeleton
# Reference: Section 3.1, Figure 2 fileciteturn0file0
# ==========================================================
class TC_Radar(nn.Module):
    def __init__(self, in_channels=1, base_channels=32, num_classes=3):
        super(TC_Radar, self).__init__()
        # Encoder: Downsampling convs + DIFB + Transformer
        self.conv_down1 = nn.Conv3d(in_channels, base_channels, kernel_size=3, stride=(2, 2, 2),
                                    padding=1)  # Sec3.1.1 fileciteturn0file0
        self.difb1 = DIFB3D(base_channels)
        # Placeholder Transformer block (use PyTorch nn.TransformerEncoderLayer)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=base_channels,
                                                            nhead=8)  # Inspired by ViT fileciteturn0file0
        # Decoder: Upsampling + Cross-Attention + Conv
        self.up_sample = nn.Upsample(scale_factor=2, mode='trilinear',
                                     align_corners=False)  # Sec3.1.2 fileciteturn0file0
        self.cross_attn = CrossAttention3D(base_channels)
        self.classifier = nn.Conv3d(base_channels, num_classes,
                                    kernel_size=1)  # Final linear layer Sec3.1.2 fileciteturn0file0

    def forward(self, x):
        # x: (B, C, D, H, W)
        enc1 = self.conv_down1(x)
        difb1 = self.difb1(enc1)
        # Flatten spatial dims for transformer: B, N, C
        B, C, D, H, W = difb1.shape
        seq = difb1.view(B, C, -1).permute(0, 2, 1)  # (B, N, C)
        trans = self.transformer_layer(seq)  # Sec3.1.1 fileciteturn0file0
        trans = trans.permute(0, 2, 1).view(B, C, D, H, W)
        # Decoder
        up = self.up_sample(trans)
        # Prepare decoder tokens similarly
        dec_seq = up.view(B, C, -1).permute(0, 2, 1)
        enc_seq = trans.view(B, C, -1).permute(0, 2, 1)
        ca = self.cross_attn(dec_seq, enc_seq)  # Sec3.3 fileciteturn0file0
        ca = ca.permute(0, 2, 1).view(B, C, D, H * 2, W * 2)
        out = self.classifier(ca)
        return out
