import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from Retrieval.loss import ClipLoss
from .modules.config_mamba import MambaConfig
from .modules.mixer_seq_simple import MixerModel


# ===================== 你给的 EEGMamba：只做“尺寸自适配”修正 =====================

class PatchEmbedding(nn.Module):
    """
    输入:  x [B, C, L, P]
    输出:  [B, C, L, d_model]
    关键尺寸约束：
      - proj_in 的 conv 输出宽度 out_w 满足 25*out_w == d_model，否则 view 会炸
      - spectral_proj 的输入维度必须是 (P//2+1)
    """
    def __init__(self, in_dim, out_dim, d_model, seq_len):
        super().__init__()
        self.d_model = d_model

        self.positional_encoding = nn.Sequential(
            nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=(7, 7),
                      stride=(1, 1), padding=(3, 3), groups=d_model, bias=False),
        )

        self.mask_encoding = nn.Parameter(torch.zeros(in_dim), requires_grad=False)

        self.proj_in = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=25, kernel_size=(1, 49),
                      stride=(1, 25), padding=(0, 24), bias=False),
            nn.GroupNorm(5, 25),
            nn.GELU(),
        )

        # ✅ 尺寸适配：rfft 的频点数 = in_dim//2+1（原来写死 101 只适配 in_dim=200）
        self.spectral_proj = nn.Sequential(
            nn.Linear(in_dim // 2 + 1, d_model, bias=False),
            nn.Dropout(0.1),
        )

    def forward(self, x, mask=None):
        bz, ch_num, patch_num, patch_size = x.shape

        if mask is None:
            mask_x = x
        else:
            mask_x = x.clone()
            mask_x[mask == 1] = self.mask_encoding

        mask_x = rearrange(mask_x, 'b c l d -> b d c l')  # [B, P, C, L]

        time_x = rearrange(mask_x, 'b d c l -> b (c l) d').unsqueeze(1)  # [B,1,C*L,P]
        time_emb = self.proj_in(time_x)  # [B,25,C*L,out_w]

        out_w = time_emb.shape[-1]
        if 25 * out_w != self.d_model:
            raise RuntimeError(f"[PatchEmbedding] need 25*out_w == d_model, "
                               f"got out_w={out_w}, 25*out_w={25*out_w}, d_model={self.d_model}. "
                               f"Try set d_model = 25*out_w (depends on in_dim={patch_size}).")

        time_emb = time_emb.permute(0, 2, 1, 3).contiguous().view(bz, ch_num, patch_num, self.d_model)

        freq_x = rearrange(mask_x, 'b d c l -> b c l d')  # [B,C,L,P]
        spectral = torch.fft.rfft(freq_x, dim=-1, norm='forward')
        spectral = torch.abs(spectral)  # [B,C,L,P//2+1]
        spectral_emb = self.spectral_proj(spectral)  # [B,C,L,d_model]

        patch_emb = time_emb + spectral_emb

        positional_embedding = self.positional_encoding(patch_emb.permute(0, 3, 1, 2))
        positional_embedding = positional_embedding.permute(0, 2, 3, 1)

        patch_emb = patch_emb + positional_embedding
        return patch_emb


def _weights_init(m):
    # 尽量不动你原逻辑，只补一个 Conv2d（否则你现在 Conv2d 没初始化）
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    if isinstance(m, (nn.Conv1d, nn.Conv2d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class EEGMambaCore(nn.Module):
    """
    这是“真正的 EEGMamba”，输出 [B,C,L,out_dim]
    注意：为了兼容 ATMS_retrieval_metrics.py 的参数统计，这个类里要有 .encoder (MixerModel)
    """
    def __init__(self, in_dim=250, out_dim=250, d_model=250, dim_feedforward=800, seq_len=1, n_layer=12):
        super().__init__()
        self.patch_embedding = PatchEmbedding(in_dim, out_dim, d_model, seq_len)

        config = MambaConfig()

        # ✅ 尺寸适配（非常关键）：你原实现没把传入 d_model/n_layer 写回 config，会导致实际跑默认值
        config.d_model = d_model
        config.n_layer = n_layer
        if hasattr(config, "d_intermediate"):
            config.d_intermediate = dim_feedforward

        config.ssm_cfg = {
            "layer": "Mamba2",
            "headdim": 50,
            "d_state": 64,
        }

        # 这里名字必须叫 encoder，方便脚本里 eeg_model.encoder.encoder 统计 backbone
        self.encoder = MixerModel(
            d_model=config.d_model,
            n_layer=config.n_layer,
            d_intermediate=config.d_intermediate,
            ssm_cfg=config.ssm_cfg,
            attn_layer_idx=config.attn_layer_idx,
            attn_cfg=config.attn_cfg,
            rms_norm=config.rms_norm,
            initializer_cfg=None,
            fused_add_norm=config.fused_add_norm,
            residual_in_fp32=config.residual_in_fp32,
        )

        self.proj_out = nn.Sequential(
            nn.Linear(d_model, out_dim),
        )
        self.apply(_weights_init)

    def forward(self, x, mask=None):
        # x: [B, C, L, P]
        bz, ch_num, seq_len, patch_size = x.shape
        hidden_states = self.patch_embedding(x, mask=mask)           # [B,C,L,d_model]
        hidden_states = rearrange(hidden_states, 'b c l d -> b (c l) d')
        hidden_states = self.encoder(hidden_states)                   # [B,C*L,d_model]
        hidden_states = rearrange(hidden_states, 'b (c l) d -> b c l d', l=seq_len)
        out = self.proj_out(hidden_states)                            # [B,C,L,out_dim]
        return out


# ===================== 给 ATMS_retrieval_metrics.py 用的“完全替换 ATMS”模型 =====================

class ATMS_EEGMamba(nn.Module):
    """
    drop-in replacement for ATMS in ATMS_retrieval_metrics.py

    约束：
      - forward(x, subject_ids) -> [B,1024]
      - 有 logit_scale / loss_func（与原脚本一致）
      - 有 self.encoder 和 self.encoder.encoder（用于参数统计）
    """
    def __init__(self,
                 num_channels=63,
                 sequence_length=250,
                 proj_dim=1024,
                 d_model=250,
                 out_dim=250,
                 n_layer=12,
                 dim_feedforward=800,
                 drop=0.1):
        super().__init__()

        # 这一层叫 encoder，脚本会统计 encoder_params
        # 且 encoder 里必须有 encoder（MixerModel），脚本会统计 backbone_params
        self.encoder = EEGMambaCore(
            in_dim=sequence_length,
            out_dim=out_dim,
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            seq_len=1,          # 适配输入 [B,63,250] -> [B,63,1,250]
            n_layer=n_layer,
        )

        # pooling + projection：把 [B,63,1,out_dim] -> [B,1024]
        self.head = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.LayerNorm(proj_dim),
        )

        # 这两个保持与你原 ATMS 完全一致
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = ClipLoss()

    def forward(self, x, subject_ids):
        """
        x: 你的 dataloader 里 eeg_data，一般是 [B,63,250]
        subject_ids: 脚本会传入，但这里不强制使用（你要用也可以以后加 subject embedding）
        """
        if x.dim() != 3:
            raise ValueError(f"Expected x with shape [B,C,T] or [B,T,C], got {x.shape}")

        # 兼容有人在别处 permute： [B,250,63] -> [B,63,250]
        if x.shape[1] == 250 and x.shape[2] == 63:
            x = x.permute(0, 2, 1).contiguous()

        B, C, T = x.shape
        if T != 250:
            raise ValueError(f"Expected T=250 (sequence_length). Got T={T}. "
                             f"Update sequence_length/in_dim/spectral_proj accordingly.")

        # 变成 EEGMamba 需要的 4D: [B,C,L,P]，这里 L=1, P=250
        x4 = x.unsqueeze(2)  # [B,63,1,250]

        feat = self.encoder(x4, mask=None)  # [B,63,1,out_dim]

        # mean pool over channel + L -> [B,out_dim]
        pooled = feat.mean(dim=(1, 2))

        out = self.head(pooled)  # [B,1024]
        return out
