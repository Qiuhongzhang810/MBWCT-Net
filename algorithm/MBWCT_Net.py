"""
1d multi channel CNN-Transformer model, pytorch
 
Qiuhong Zhang, Nov 2026
MBWCT-Net (Multi-Branch Wavelet-CNN Transformer Network)
"""

import numpy as np
from collections import Counter
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report 

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# 添加了
import torch
import torch.nn as nn
from pytorch_wavelets import DWT1DForward  # pip install pytorch_wavelets==1.3.0
import math

class WaveletFeature1D(nn.Module):
    def __init__(self, wave='db4', J=1, out_features=None):
        """
        wave: 小波类型
        J: 分解层数
        out_features: 最终要输出的特征长度（可选，若需降维）
        """
        super().__init__()
        self.dwt = DWT1DForward(J=J, mode='zero', wave=wave)
        self.out_features = out_features

    def forward(self, x):
        # x: (batch, channels, length)
        yl, yh = self.dwt(x)  # yl: 低频, yh: [每层的高频]
        features = [yl]
        for h in yh:  # 每一层的高频
            features.append(h)
        x_feat = torch.cat(features, dim=1)  # 按channel拼接
        # Optionally flatten or apply 1x1卷积等处理
        if self.out_features is not None:
            # 用1x1卷积将特征数调整到out_features
            conv = nn.Conv1d(x_feat.size(1), self.out_features, 1).to(x_feat.device)
            x_feat = conv(x_feat)
        return x_feat  # 输出: (batch, out_channels, new_length)


class MyDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        return (torch.tensor(self.data[index], dtype=torch.float), torch.tensor(self.label[index], dtype=torch.long))

    def __len__(self):
        return len(self.data)

class ECAAttention(nn.Module):
    def __init__(self, channel, k_size=3):
        super(ECAAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # 修改为1D平均池化
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 平均池化后需要调整维度以符合一维卷积的要求
        y = self.avg_pool(x).squeeze(-1)  # B x C x 1
        # 因为Conv1d期望的输入是(B, C, L)，我们需要转换维度以匹配
        y = self.conv(y.unsqueeze(1)).squeeze(1)  # 压缩最后一维，然后调整维度以适应1D卷积
        y = self.sigmoid(y)  # 应用sigmoid激活函数
        return x * y.unsqueeze(-1)  # 扩展y并与输入x相乘，实现通道注意力

# ========== MLCA 注意力模块（可二次创新） ==========
class MLCA(nn.Module):
    def __init__(self, in_size, local_size=5, gamma=2, b=1, local_weight=0.5):
        super().__init__()
        self.local_size = local_size
        self.gamma = gamma
        self.b = b
        t = int(abs(math.log(in_size, 2) + self.b) / self.gamma)
        k = t if t % 2 else t + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.conv_local = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        # 动态local_weight: 每个通道一个可学习参数
        self.local_weight = nn.Parameter(torch.ones(in_size) * local_weight)
        self.local_arv_pool = nn.AdaptiveAvgPool2d(local_size)
        self.global_arv_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        local_arv = self.local_arv_pool(x)
        global_arv = self.global_arv_pool(local_arv)
        b, c, m, n = x.shape
        b_local, c_local, m_local, n_local = local_arv.shape
        temp_local = local_arv.view(b, c_local, -1).transpose(-1, -2).reshape(b, 1, -1)
        temp_global = global_arv.view(b, c, -1).transpose(-1, -2)
        y_local = self.conv_local(temp_local)
        y_global = self.conv(temp_global)
        y_local_transpose = y_local.reshape(b, self.local_size * self.local_size, c).transpose(-1, -2).view(b, c, self.local_size, self.local_size)
        y_global_transpose = y_global.view(b, -1).unsqueeze(-1).unsqueeze(-1)
        att_local = y_local_transpose.sigmoid()
        att_global = F.adaptive_avg_pool2d(y_global_transpose.sigmoid(), [self.local_size, self.local_size])
        # 动态适配local_weight
        in_channels = x.shape[1]
        if self.local_weight.shape[0] != in_channels:
            local_weight = torch.ones(in_channels, device=x.device, dtype=x.dtype) * 0.5
        else:
            local_weight = self.local_weight
        att_all = F.adaptive_avg_pool2d(
            att_global * (1 - local_weight.view(1, -1, 1, 1)) + (att_local * local_weight.view(1, -1, 1, 1)),
            [m, n]
        )
        x = x * att_all
        return x

# ========== 轻量化 Cross-Attention 模块 ==========
class LightCrossAttention(nn.Module):
    def __init__(self, num_branches, channels):
        super().__init__()
        self.query = nn.Linear(channels, channels // 2)
        self.key = nn.Linear(channels, channels // 2)
        self.value = nn.Linear(channels, channels)
        self.softmax = nn.Softmax(dim=-1)
        self.num_branches = num_branches
        self.channels = channels

    def forward(self, branch_feats):
        # branch_feats: list of [B, C, L]，先堆叠
        x = torch.stack(branch_feats, dim=1)  # [B, num_branches, C, L]
        B, N, C, L = x.shape
        x_flat = x.permute(0, 1, 3, 2).reshape(B*N*L, C)  # [B*N*L, C]
        Q = self.query(x_flat).view(B, N, L, -1)  # [B, N, L, C//2]
        K = self.key(x_flat).view(B, N, L, -1)
        V = self.value(x_flat).view(B, N, L, C)
        # 只做分支间注意力（对每个位置L独立）
        attn = torch.einsum('bnlc,bmlc->bnml', Q, K) / (Q.shape[-1] ** 0.5)  # [B, N, N, L]
        attn = self.softmax(attn)
        out = torch.einsum('bnml,bmlc->bnlc', attn, V)  # [B, N, L, C]
        out = out.permute(0, 1, 3, 2).reshape(B, N, C, L)
        # 融合所有分支（可sum/mean/concat+1x1conv）
        fused = out.sum(dim=1)  # [B, C, L]
        return fused

# ========== MSAT-Former: 多尺度自适应Transformer (创新设计) ==========
class MSATFormer(nn.Module):
    """
    Multi-Scale Adaptive Transformer (MSAT-Former)
    
    创新点：
    1. 融合局部窗口注意力和全局自注意力
    2. 可学习的通道级自适应融合权重
    3. 内置残差连接和层归一化
    4. 多尺度特征提取（类似MLCA但用Transformer实现）
    
    替代原来的：MLCA + Transformer + 残差连接
    """
    def __init__(self, d_model=64, nhead=8, num_layers=1, dim_feedforward=128, 
                 dropout=0.3, window_size=5, local_weight=0.5):
        super().__init__()
        self.d_model = d_model
        self.window_size = window_size
        
        # 局部窗口自注意力（用于捕获局部模式）
        self.local_attention = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=nhead, 
            dropout=dropout,
            batch_first=True  # 改为batch_first=True以简化处理
        )
        
        # 全局自注意力（用于捕获长程依赖）
        self.global_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True  # 改为batch_first=True以简化处理
        )
        
        # 可学习的通道级融合权重（每个通道独立学习局部/全局平衡）
        self.local_weight = nn.Parameter(torch.ones(d_model) * local_weight)
        
        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def _local_window_attention(self, x):
        """
        局部窗口注意力：将序列分成多个窗口，在每个窗口内计算注意力
        x: (B, L, C)
        """
        B, L, C = x.shape
        window_size = min(self.window_size, L)
        
        # 如果序列长度小于窗口大小，直接使用全局注意力
        if L <= window_size:
            return self.global_attention(x, x, x)[0]
        
        # 将序列分成多个重叠窗口
        num_windows = (L + window_size - 1) // window_size
        outputs = []
        
        for i in range(num_windows):
            start = i * window_size
            end = min(start + window_size, L)
            window_x = x[:, start:end, :]  # (B, window_size, C)
            
            # 窗口内自注意力
            window_out, _ = self.local_attention(window_x, window_x, window_x)
            outputs.append(window_out)
        
        # 拼接所有窗口输出
        local_out = torch.cat(outputs, dim=1)  # (B, L, C)
        return local_out
    
    def forward(self, x):
        """
        x: (B, C, L) - 输入特征
        返回: (B, C, L) - 输出特征
        """
        B, C, L = x.shape
        
        # 转换为Transformer输入格式: (B, L, C)
        x_trans = x.permute(0, 2, 1)  # (B, L, C)
        residual = x_trans
        
        # ========== 第一阶段：局部窗口注意力 ==========
        local_out = self._local_window_attention(x_trans)
        local_out = self.norm1(local_out + residual)  # 残差连接 + 层归一化
        
        # ========== 第二阶段：全局自注意力 ==========
        global_out, _ = self.global_attention(x_trans, x_trans, x_trans)
        global_out = self.norm2(global_out + residual)  # 残差连接 + 层归一化
        
        # ========== 第三阶段：自适应融合局部和全局特征 ==========
        # 可学习的通道级权重
        alpha = torch.sigmoid(self.local_weight)  # (C,)
        alpha = alpha.view(1, 1, -1)  # (1, 1, C) 用于广播
        
        # 自适应融合：每个通道独立平衡局部和全局信息
        fused = alpha * local_out + (1 - alpha) * global_out  # (B, L, C)
        fused = self.dropout(fused)
        
        # ========== 第四阶段：Feed-Forward Network ==========
        ffn_out = self.ffn(fused)
        output = self.norm3(ffn_out + fused)  # 残差连接 + 层归一化
        
        # 转换回原始格式: (B, C, L)
        output = output.permute(0, 2, 1)  # (B, C, L)
        
        return output

# ========== MBWCT-Net 主网络结构 ==========
class MBWCTNet(nn.Module):
    def __init__(self, in_channels, n_len_seg, n_classes, device, num_branches=5, wave='db4', J=2, verbose=False,
                 use_wavelet=True, use_cross_attn=True, use_mlca=True, use_residual=True, use_msat_former=False):
        super().__init__()
        self.n_len_seg = n_len_seg
        self.n_classes = n_classes
        self.in_channels = in_channels
        self.device = device
        self.verbose = verbose
        self.num_branches = num_branches
        self.wavelet = DWT1DForward(J=J, mode='zero', wave=wave)
        self.use_wavelet = use_wavelet
        self.use_cross_attn = use_cross_attn
        self.use_mlca = use_mlca
        self.use_residual = use_residual
        self.use_msat_former = use_msat_former  # 是否使用新的MSAT-Former
        
        # 分支卷积+注意力
        self.branch_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, 32, kernel_size=7, stride=1, padding=3),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Conv1d(32, 32, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(n_len_seg),
            ) for _ in range(num_branches)
        ])
        self.branch_attn = nn.ModuleList([
            ECAAttention(64) for _ in range(num_branches)
        ])
        self.cross_attn = LightCrossAttention(num_branches=num_branches, channels=64)
        
        # 根据是否使用MSAT-Former选择不同的结构
        if use_msat_former:
            # 使用新的MSAT-Former（融合MLCA+Transformer+残差）
            self.msat_former = MSATFormer(
                d_model=64, 
                nhead=8, 
                num_layers=1, 
                dim_feedforward=128, 
                dropout=0.3,
                window_size=5,
                local_weight=0.5
            )
            # 如果使用MSAT-Former，不需要单独的MLCA和Transformer
            self.mlca = None
            self.transformer_encoder = None
        else:
            # 使用原来的MLCA + Transformer + 残差结构
            # 动态设置MLCA的in_size
            if use_cross_attn and num_branches > 1:
                mlca_in_size = 64
            else:
                mlca_in_size = 64 * num_branches if num_branches > 1 else 64
            self.mlca = MLCA(in_size=mlca_in_size, local_size=5)
            encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=8, dim_feedforward=128, dropout=0.3, batch_first=True)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
            self.msat_former = None
        
        # 拼接分支时加1x1卷积统一通道数
        self.fuse_proj = None
        if not use_cross_attn and num_branches > 1:
            self.fuse_proj = nn.Conv1d(64 * num_branches, 64, kernel_size=1)
        self.dense = nn.Linear(64, n_classes)

    def forward(self, x):
        B, C, L = x.shape
        if self.use_wavelet:
            yl, yh = self.wavelet(x)
            branches = [yl] + [h for h in yh[:self.num_branches-1]]
        else:
            branches = [x]
        branch_outs = []
        for i, feat in enumerate(branches[:len(self.branch_convs)]):
            out = self.branch_convs[i](feat)
            out = self.branch_attn[i](out)
            branch_outs.append(out)
        if self.use_cross_attn and len(branch_outs) > 1:
            fused = self.cross_attn(branch_outs)
        else:
            fused = torch.cat(branch_outs, dim=1) if len(branch_outs) > 1 else branch_outs[0]
            # 只要通道数不是64就动态映射到64
            if fused.shape[1] != 64 and self.fuse_proj is not None:
                fused = self.fuse_proj(fused)
            elif fused.shape[1] != 64:
                # 动态投影
                fuse_proj_dynamic = nn.Conv1d(fused.shape[1], 64, kernel_size=1).to(fused.device)
                fused = fuse_proj_dynamic(fused)
        # ========== 使用MSAT-Former（创新结构）或原来的MLCA+Transformer+残差 ==========
        if self.use_msat_former:
            # 使用新的MSAT-Former：融合局部窗口注意力、全局自注意力和自适应融合
            # 确保通道数为64
            if fused.shape[1] != 64:
                if self.fuse_proj is not None:
                    fused = self.fuse_proj(fused)
                else:
                    fuse_proj_dynamic = nn.Conv1d(fused.shape[1], 64, kernel_size=1).to(fused.device)
                    fused = fuse_proj_dynamic(fused)
            # MSAT-Former内部已经包含了局部注意力、全局注意力、自适应融合和残差连接
            out = self.msat_former(fused)  # (B, 64, L)
        else:
            # 使用原来的结构：MLCA + Transformer + 残差连接
            fused_2d = fused.unsqueeze(-1)
            if self.use_mlca:
                mlca_out = self.mlca(fused_2d).squeeze(-1)
            else:
                mlca_out = fused
            # 送入Transformer前强制对齐通道数为64
            if mlca_out.shape[1] != 64:
                if self.fuse_proj is not None:
                    mlca_out = self.fuse_proj(mlca_out)
                else:
                    fuse_proj_dynamic2 = nn.Conv1d(mlca_out.shape[1], 64, kernel_size=1).to(mlca_out.device)
                    mlca_out = fuse_proj_dynamic2(mlca_out)
            trans_out = self.transformer_encoder(mlca_out.transpose(-1, -2))  # 使用batch_first
            trans_out = trans_out.transpose(-1, -2)
            if self.use_residual and self.use_mlca:
                out = mlca_out + trans_out
            else:
                out = trans_out
        
        # 全局平均池化 + 分类
        out = out.mean(-1)  # (B, 64)
        out = self.dense(out)  # (B, n_classes)
        return out