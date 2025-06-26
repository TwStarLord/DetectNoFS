# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# # ================================================
# # 前置：Mamba SSM 官方实现依赖安装
# # pip install mamba-ssm[causal-conv1d]
# # 来自 state-spaces/mamba 仓库 ([github.com](https://github.com/state-spaces/mamba))
# from mamba_ssm import Mamba
#
# """
# 总体说明：
# 本文件集成四种网络结构：
# 1. 基于官方 Mamba SSM 模块的时序特征提取
# 2. 使用 CBAM 替换 DIFB 的图像注意力模块
# 3. 双分支多模态融合网络（时域+时频图）
# 4. 单模态多特征融合网络（时序+图像统计量）
#
# 各模块原理及公式概述：
# (1) Mamba SSM:
#     使用选择性状态空间模型(SSM)和局部卷积捕获长短程依赖。
#     X_proj = W_x X,  Y = Mamba(X_proj) ∈ ℝ^{B×T×d_model} ([arxiv.org](https://arxiv.org/abs/2312.00752?utm_source=chatgpt.com), [github.com](https://github.com/state-spaces/mamba))
#     Mamba 内部：SSM 生成全局记忆 h, 局部卷积生成 X_local, 残差融合 + 层归一化 + MLP。
# (2) CBAM 注意力：通道注意力 M_c 和空间注意力 M_s，对输入特征加权 ([zh.wikipedia.org](https://zh.wikipedia.org/wiki/Mamba_%28%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%9E%B6%E6%9E%84%29?utm_source=chatgpt.com), [en.wikipedia.org](https://en.wikipedia.org/wiki/Mamba_%28deep_learning_architecture%29?utm_source=chatgpt.com))
# (3) 双分支融合：f_seq = Mamba(x_seq), f_img = CNN(x_img), f = [f_seq; f_img] → FC
# (4) 多特征融合：序列统计μ, σ与图像统计拼接 → FC
# """
#
# # ================================================
# # 1. 基于官方 Mamba SSM 的时序网络
# # ================================================
# class ModelMambaSSM(nn.Module):
#     def __init__(self, in_ch, d_model, d_state, d_conv, expand, num_classes):
#         super().__init__()
#         # 输入线性投影到 Mamba 维度
#         self.proj = nn.Linear(in_ch, d_model)
#         # Mamba 块：使用官方包 mamba_ssm.Mamba ([github.com](https://github.com/state-spaces/mamba))
#         self.mamba = Mamba(
#             d_model=d_model,    # 模型维度
#             d_state=d_state,    # 状态扩展因子
#             d_conv=d_conv,      # 局部卷积宽度
#             expand=expand      # 块扩展因子
#         )
#         # 分类器：在每个时间步输出分类结果
#         self.classifier = nn.Linear(d_model, num_classes)
#
#     def forward(self, x):  # x: (B, T, in_ch)
#         # 投影
#         x = self.proj(x)             # (B, T, d_model)
#         # Mamba SSM 计算 (保留所有时间步)
#         y = self.mamba(x)            # (B, T, d_model)
#         # 每时间步分类
#         out = self.classifier(y)     # (B, T, num_classes)
#         return out
#
# # ================================================
# # 2. CBAM 注意力模块
# # ================================================
# class CBAM(nn.Module):
#     def __init__(self, channels, reduction=16, kernel_size=7):
#         super().__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#         self.mlp      = nn.Sequential(
#             nn.Linear(channels, channels//reduction, bias=False),
#             nn.ReLU(),
#             nn.Linear(channels//reduction, channels, bias=False)
#         )
#         self.conv     = nn.Conv2d(2,1,kernel_size,padding=kernel_size//2,bias=False)
#     def forward(self, x):
#         B,C,H,W = x.shape
#         avg = self.avg_pool(x).view(B,C)
#         max_ = self.max_pool(x).view(B,C)
#         ca = torch.sigmoid(self.mlp(avg)+self.mlp(max_)).view(B,C,1,1)
#         x_ca = x * ca
#         avg_sp = x_ca.mean(dim=1, keepdim=True)
#         max_sp,_ = x_ca.max(dim=1, keepdim=True)
#         sa = torch.sigmoid(self.conv(torch.cat([avg_sp, max_sp], dim=1)))
#         return x_ca * sa
#
# class ModelCBAM(nn.Module):
#     def __init__(self, in_ch, num_classes):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_ch, 64, 3, padding=1)
#         self.cbam  = CBAM(64)
#         self.pool  = nn.AdaptiveAvgPool2d((1,1))
#         self.fc    = nn.Linear(64, num_classes)
#     def forward(self, x):  # x: (B, C, H, W)
#         x = F.relu(self.conv1(x))
#         x = self.cbam(x)
#         x = self.pool(x).view(x.size(0), -1)
#         return self.fc(x)
#
# # ================================================
# # 3. 双分支多模态网络
# # ================================================
# class DualBranchModal(nn.Module):
#     def __init__(self, seq_in, img_ch, d_model, d_state, d_conv, expand, hidden_dim, num_classes):
#         super().__init__()
#         # 时域分支
#         self.seq_net = ModelMambaSSM(seq_in, d_model, d_state, d_conv, expand, hidden_dim)
#         # 图像分支
#         self.img_net = nn.Sequential(
#             nn.Conv2d(img_ch, 32, 3, padding=1),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool2d((4,4))
#         )
#         # 融合分类
#         self.cls = nn.Sequential(
#             nn.Linear(hidden_dim + 32*4*4, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, num_classes)
#         )
#     def forward(self, x_seq, x_img):
#         f_seq = self.seq_net(x_seq)         # (B, T, hidden_dim)
#         f_seq = f_seq[:, -1, :]             # 仅取最后时间步表示
#         img_feat = self.img_net(x_img).view(x_img.size(0), -1)
#         f = torch.cat([f_seq, img_feat], dim=1)
#         return self.cls(f)
#
# # ================================================
# # 4. 单模态多特征融合
# # ================================================
# class FeatureFusionSingle(nn.Module):
#     def __init__(self, seq_in, d_model, d_state, d_conv, expand, hidden_dim, num_classes):
#         super().__init__()
#         self.seq_proj = ModelMambaSSM(seq_in, d_model, d_state, d_conv, expand, hidden_dim)
#         self.stat_fc  = nn.Linear(2, hidden_dim)
#         self.cls      = nn.Sequential(
#             nn.Linear(hidden_dim*2, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, num_classes)
#         )
#     def forward(self, x_seq, x_img):
#         seq_out = self.seq_proj(x_seq)     # (B, T, hidden_dim)
#         seq_feat = seq_out[:, -1, :]
#         B,C,H,W = x_img.shape
#         flat = x_img.view(B, -1)
#         stat = torch.stack([flat.mean(dim=1), flat.std(dim=1)], dim=1)
#         stat_feat = F.relu(self.stat_fc(stat))
#         f = torch.cat([seq_feat, stat_feat], dim=1)
#         return self.cls(f)
#
# # ================================================
# # 示例调用与性能测试接口
# # ================================================
# if __name__ == "__main__":
#     import time
#     from sklearn.metrics import accuracy_score, precision_recall_fscore_support  # 需安装scikit-learn
#     B,T,C,H,W = 16, 1024, 2, 64, 64
#     x_seq = torch.randn(B, T, C).cuda()
#     x_img = torch.randn(B, C, H, W).cuda()
#     model = ModelMambaSSM(C, 128, 64, 4, 2, 10).cuda()
#     # 训练速度测试
#     start = time.time()
#     y = model(x_seq)
#     torch.cuda.synchronize()
#     end = time.time()
#     print(f"MambaSSM 执行时间: {end-start:.4f}s")
#     # 模型参数量
#     p = sum(p.numel() for p in model.parameters())
#     print(f"模型参数量: {p}")
#     # 长序列建模：输出维度验证
