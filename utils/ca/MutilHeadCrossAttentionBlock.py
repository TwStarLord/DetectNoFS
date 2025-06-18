from torch import nn


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