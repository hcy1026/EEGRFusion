import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F


from Retrieval.loss import ClipLoss
import numpy as np
# 你需要补上（或确认你工程里存在）：
from .modules.model_neural_transformer import NTConfig, NeuralTransformer
from .modules.model import *


class _NeuroLMBackbone(nn.Module):
    """
    兼容你脚本的参数统计：
      - eeg_model.encoder         -> backbone
      - eeg_model.encoder.encoder -> NeuralTransformer
    """
    def __init__(
        self,
        # ====== 你要的默认尺寸 ======
        num_channels: int = 63,
        sequence_length: int = 250,
        d_model: int = 250,              # <- 你说的 dmodal250
        # ====== Transformer 结构默认 ======
        n_layer: int = 12,
        n_head: int = 10,                # 250/10=25 每头维度整数
        block_size: int = 1024,
        bias: bool = False,
        dropout: float = 0.0,
        in_chans: int = 1,
        out_chans: int = 16,
        # ====== 其他 ======
        max_subjects: int = 256,
        tokenizer_ckpt_path: str = None,
        freeze_tokenizer: bool = True,
        use_subject_embed: bool = True,
        force_input_len: bool = True,    # 强制把输入插值到 250，避免 pipeline 里冒出 800
    ):
        super().__init__()
        self.num_channels = num_channels
        self.sequence_length = sequence_length
        self.hidden_dim = d_model
        self.max_subjects = max_subjects
        self.use_subject_embed = use_subject_embed
        self.force_input_len = force_input_len

        if d_model % n_head != 0:
            raise ValueError(f"d_model({d_model}) must be divisible by n_head({n_head}).")

        # 1) NeuralTransformer config：默认 n_embd=d_model=250
        encoder_args = dict(
            n_layer=n_layer,
            n_head=n_head,
            n_embd=d_model,
            block_size=block_size,
            bias=bias,
            dropout=dropout,
            num_classes=0,
            in_chans=in_chans,
            out_chans=out_chans,
            patch_size=sequence_length,   # 只是记录用；TemporalConv 不直接用它
        )
        enc_conf = NTConfig(**encoder_args)

        # 2) tokenizer（必须挂在 .encoder 这个名字下，兼容 eeg_model.encoder.encoder）
        self.encoder = NeuralTransformer(enc_conf)

        # ====== 关键：把 TemporalConv.l 改成“确定形状”的 Linear(in_features=?, out_features=d_model) ======
        # TemporalConv.forward: conv 输出 [B, out_chans, NA, L] -> rearrange -> [B, NA, (L*out_chans)]
        # 所以 Linear 的 in_features = L * out_chans，其中 L 是 conv1 后的时间长度
        if hasattr(self.encoder.patch_embed, "conv1") and hasattr(self.encoder.patch_embed, "l"):
            conv1 = self.encoder.patch_embed.conv1

            # PyTorch Conv 输出长度公式：
            # L_out = floor((L_in + 2p - d*(k-1) - 1)/s + 1)
            k = conv1.kernel_size[1]
            s = conv1.stride[1]
            p = conv1.padding[1]
            d = conv1.dilation[1]
            L_out = math.floor((sequence_length + 2*p - d*(k-1) - 1) / s + 1)

            in_features = L_out * out_chans  # out_chans = conv1.out_channels
            self.encoder.patch_embed.l = nn.Sequential(
                nn.Linear(in_features, d_model),
                nn.GELU()
            )
            print(f"[NeuroLM] patch_embed Linear set to: in_features={in_features} -> d_model={d_model} "
                  f"(seq_len={sequence_length}, L_out={L_out}, out_chans={out_chans})")

        # 4) 可选：加载 tokenizer checkpoint（如果你以后要用预训练 tokenizer）
        # 注意：用了 LazyLinear 后，ckpt 的 Linear 权重形状通常对不上，所以：
        # - 如果你要 load ckpt：先把上面 LazyLinear 注释掉，再按 ckpt 期望的 in_features 固定 Linear
        if tokenizer_ckpt_path is not None:
            ckpt = torch.load(tokenizer_ckpt_path, map_location="cpu")
            sd = ckpt.get("model", ckpt)

            unwanted_prefix = "_orig_mod."
            for k, v in list(sd.items()):
                if k.startswith(unwanted_prefix):
                    sd[k[len(unwanted_prefix):]] = sd.pop(k)

            new_sd = OrderedDict()
            for k in list(sd.keys()):
                if k.startswith("VQ.encoder."):
                    new_sd[k[11:]] = sd[k]

            missing, unexpected = self.encoder.load_state_dict(new_sd, strict=False)
            print(f"[NeuroLMBackbone] loaded tokenizer ckpt: {tokenizer_ckpt_path}")
            if missing:
                print("[NeuroLMBackbone] missing keys:", missing[:20])
            if unexpected:
                print("[NeuroLMBackbone] unexpected keys:", unexpected[:20])

        # 5) 冻结 tokenizer（默认冻结，先跑通）
        if freeze_tokenizer:
            for p in self.encoder.parameters():
                p.requires_grad = False

        # 6) subject embedding（为了不改训练代码：forward(x, subject_ids)）
        self.subject_embed = nn.Embedding(max_subjects, d_model) if use_subject_embed else None

    def _resize_time(self, x: torch.Tensor, target_T: int) -> torch.Tensor:
        """
        x: [B, C, T] -> [B, C, target_T] via 1D linear interpolation
        """
        B, C, T = x.shape
        if T == target_T:
            return x
        x_ = x.reshape(B * C, 1, T)  # [BC,1,T]
        x_ = F.interpolate(x_, size=target_T, mode="linear", align_corners=False)
        return x_.reshape(B, C, target_T)

    def forward(self, x, subject_ids=None):
        """
        x: [B, C, T] 或 [B, 1, C, T]
        return: token features [B, C, d_model]
        """
        if x.dim() == 4:
            x = x.squeeze(1)  # [B,C,T]
        if x.dim() != 3:
            raise ValueError(f"Expected EEG shape [B,C,T] or [B,1,C,T], got {tuple(x.shape)}")

        B, C, T = x.shape
        device = x.device

        # 强制对齐到你原始输入长度 250，避免 pipeline 里出现 800
        if self.force_input_len and (T != self.sequence_length):
            x = self._resize_time(x, self.sequence_length)
            T = self.sequence_length

        # NeuralTransformer 内部会用 input_chans/input_times 做 embedding
        input_chans = torch.arange(C, device=device).unsqueeze(0).repeat(B, 1).long()
        input_time = torch.zeros((B, C), device=device).long()

        # 关键：mask 传 None（你工程这版 SDPA 期望 token×token mask，不要传 [B,1,C,T]）
        feats = self.encoder(x, input_chans, input_time, mask=None, return_all_tokens=True)  # [B,C,d_model]

        # + subject embedding（可选）
        if (self.subject_embed is not None) and (subject_ids is not None):
            sid = torch.clamp(subject_ids.long(), 0, self.max_subjects - 1)
            feats = feats + self.subject_embed(sid).unsqueeze(1)  # [B,1,d_model]

        return feats


class NeuroLM(nn.Module):
    """
    Drop-in 替换 ATMS EEG encoder。
    默认：
      num_channels=63, sequence_length=250, d_model=250, proj_dim=1024
    forward(x, subject_ids) -> [B,1024]
    """
    def __init__(
        self,
        num_channels: int = 63,
        sequence_length: int = 250,
        d_model: int = 250,
        proj_dim: int = 1024,
        tokenizer_ckpt_path: str = None,
        freeze_tokenizer: bool = True,
        pool: str = "mean",        # "mean" or "last"
        drop_proj: float = 0.5,
    ):
        super().__init__()

        self.encoder = _NeuroLMBackbone(
            num_channels=num_channels,
            sequence_length=sequence_length,
            d_model=d_model,
            tokenizer_ckpt_path=tokenizer_ckpt_path,
            freeze_tokenizer=freeze_tokenizer,
            force_input_len=True,
        )

        if pool not in ["mean", "last"]:
            raise ValueError(f"pool must be 'mean' or 'last', got {pool}")
        self.pool = pool

        # projection 到 CLIP dim=1024
        self.proj_eeg = nn.Sequential(
            nn.Linear(d_model, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
            nn.Dropout(drop_proj),
            nn.LayerNorm(proj_dim),
        )

        # 训练脚本依赖
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = ClipLoss()

    def forward(self, x, subject_ids):
        feats = self.encoder(x, subject_ids=subject_ids)  # [B,C,d_model]
        if self.pool == "mean":
            pooled = feats.mean(dim=1)      # [B,d_model]
        else:
            pooled = feats[:, -1, :]        # [B,d_model]
        out = self.proj_eeg(pooled)         # [B,1024]
        return out

