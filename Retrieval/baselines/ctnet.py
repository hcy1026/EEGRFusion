# ctnet.py
import math
import argparse
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 你的脚本里就是 from loss import ClipLoss
from Retrieval.loss import ClipLoss
from .modules.ctnetmodule import CTNet as _CTNet

class _CTNetBackbone250(nn.Module):
    """
    只是为了兼容你 main() 里这两句统计：
        encoder_params = count_params(eeg_model.encoder, ...)
        backbone_params = count_params(eeg_model.encoder.encoder, ...)
    所以这里做一个“有 .encoder 属性”的 wrapper：
        - self.backbone: CTNet 整体
        - self.encoder : 仅 transformer 部分 (trans)，用于 backbone_params 统计
    """
    def __init__(
        self,
        num_channels: int = 63,
        sequence_length: int = 250,
        feature_dim: int = 250,     # 你说的“给定250”
        # CTNet结构超参：全部显式给定，避免 _resolve_dims 自动推断
        embed_dim: int = 40,
        n_filters_time: int = 20,   # 40 = 2 * 20
        depth_multiplier: int = 2,
        num_heads: int = 4,
        num_layers: int = 6,
        kernel_size: int = 64,
        pool_size_1: int = 8,
        pool_size_2: int = 8,
        cnn_drop_prob: float = 0.3,
        att_positional_drop_prob: float = 0.1,
        final_drop_prob: float = 0.5,
    ):
        super().__init__()

        # 局部 import，避免你去改 import 区
        self.backbone = _CTNet(
            n_outputs=feature_dim,      # 让 CTNet final_layer 输出 250 维
            n_chans=num_channels,
            n_times=sequence_length,

            # 显式固定，避免自动推断
            embed_dim=embed_dim,
            n_filters_time=n_filters_time,
            depth_multiplier=depth_multiplier,

            # 其余超参
            num_heads=num_heads,
            num_layers=num_layers,
            kernel_size=kernel_size,
            pool_size_1=pool_size_1,
            pool_size_2=pool_size_2,
            cnn_drop_prob=cnn_drop_prob,
            att_positional_drop_prob=att_positional_drop_prob,
            final_drop_prob=final_drop_prob,
        )

        # 兼容 main() 的 backbone_params 统计：把 transformer 部分挂到 .encoder 上
        # CTNet forward 里就是：cnn -> position -> trans -> ...
        self.encoder = self.backbone.trans

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CTNet 的 forward 就是你贴的 ensuredim/cnn/position/trans/.../final_layer
        return self.backbone(x)  # (B, 250)


class CTNet(nn.Module):
    """
    用 CTNet 做 EEG encoder，并把输出对齐到 ATMS retrieval 需要的 1024 维：
        (B,63,250) -> CTNet -> (B,250) -> proj -> (B,1024)

    兼容 train_model 的调用：
        eeg_features = eeg_model(eeg_data, subject_ids).float()
    所以 forward 必须接收 subject_ids（但这里不用它）。
    """
    def __init__(
        self,
        num_channels: int = 63,
        sequence_length: int = 250,
        feature_dim: int = 250,     # 你说的“给定250”
        proj_dim: int = 1024,       # 你说的“给定1024”
        drop_proj: float = 0.5,
        # CTNet结构超参（全部显式给定，不推断）
        embed_dim: int = 40,
        n_filters_time: int = 20,
        depth_multiplier: int = 2,
        num_heads: int = 4,
        num_layers: int = 6,
        kernel_size: int = 64,
        pool_size_1: int = 8,
        pool_size_2: int = 8,
        cnn_drop_prob: float = 0.3,
        att_positional_drop_prob: float = 0.1,
        final_drop_prob: float = 0.5,
    ):
        super().__init__()

        # 这里的 self.encoder 要兼容你 main() 的参数统计写法：
        #   encoder_params = count_params(eeg_model.encoder, ...)
        #   backbone_params = count_params(eeg_model.encoder.encoder, ...)
        self.encoder = _CTNetBackbone250(
            num_channels=num_channels,
            sequence_length=sequence_length,
            feature_dim=feature_dim,
            embed_dim=embed_dim,
            n_filters_time=n_filters_time,
            depth_multiplier=depth_multiplier,
            num_heads=num_heads,
            num_layers=num_layers,
            kernel_size=kernel_size,
            pool_size_1=pool_size_1,
            pool_size_2=pool_size_2,
            cnn_drop_prob=cnn_drop_prob,
            att_positional_drop_prob=att_positional_drop_prob,
            final_drop_prob=final_drop_prob,
        )

        # ===== 关键：把 250 -> 1024 的对齐“包进 ctnet class” =====
        self.proj_eeg = nn.Sequential(
            nn.Linear(feature_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
            nn.Dropout(drop_proj),
            nn.LayerNorm(proj_dim),
        )

        # 训练/评估用到的这俩属性必须保留（train_model/test_model 里直接读）
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = ClipLoss()

    def forward(self, x: torch.Tensor, subject_ids=None) -> torch.Tensor:
        # x: (B, 63, 250)
        out = self.encoder(x)        # (B, 250) —— CTNet forward 路径
        out = self.proj_eeg(out)     # (B, 1024) —— 对齐到 ATMS retrieval
        return out

