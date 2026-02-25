import torch
import torch.nn as nn

# 你原脚本里应该已经有 ClipLoss（ATMS 用的那个）
from Retrieval.loss import ClipLoss
from .modules.eegnexmodule import EEGNeX as _EEGNeX
import numpy as np
# 如果你脚本里 ClipLoss 路径不同，就保持你原来的 import 不动即可


class EEGNexEncoder(nn.Module):
    """
    Thin wrapper that exposes:
      - self.encoder : the underlying braindecode EEGNeX model
    so downstream can do: count_params(eeg_model.encoder.encoder)
    and still get the raw backbone params.

    Output is forced to (B, embed_dim) by setting n_outputs=embed_dim in EEGNeX.
    """
    def __init__(
        self,
        n_chans: int,
        n_times: int,
        embed_dim: int = 1024,
        sfreq: float | None = None,
        # EEGNeX hyperparams (keep defaults to minimize diffs)
        activation: type[nn.Module] = nn.ELU,
        depth_multiplier: int = 2,
        filter_1: int = 8,
        filter_2: int = 32,
        drop_prob: float = 0.5,
        kernel_block_1_2: int = 64,
        kernel_block_4: int = 16,
        dilation_block_4: int = 2,
        avg_pool_block4: int = 4,
        kernel_block_5: int = 16,
        dilation_block_5: int = 4,
        avg_pool_block5: int = 8,
        max_norm_conv: float = 1.0,
        max_norm_linear: float = 0.25,
    ):
        super().__init__()

        # IMPORTANT:
        # Use EEGNeX's built-in final_layer to output embed_dim directly.
        # This guarantees output shape (B, 1024) without modifying its forward.
        self.encoder = _EEGNeX(
            n_chans=n_chans,
            n_times=n_times,
            n_outputs=embed_dim,
            sfreq=sfreq,
            activation=activation,
            depth_multiplier=depth_multiplier,
            filter_1=filter_1,
            filter_2=filter_2,
            drop_prob=drop_prob,
            kernel_block_1_2=kernel_block_1_2,
            kernel_block_4=kernel_block_4,
            dilation_block_4=dilation_block_4,
            avg_pool_block4=avg_pool_block4,
            kernel_block_5=kernel_block_5,
            dilation_block_5=dilation_block_5,
            avg_pool_block5=avg_pool_block5,
            max_norm_conv=max_norm_conv,
            max_norm_linear=max_norm_linear,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_chans, n_times)
        return self.encoder(x)


class EEGNex(nn.Module):
    """
    Drop-in replacement for ATMS-style EEG encoder module.

    Requirements you listed:
      - minimal change, keep forward structure simple
      - output (B, 1024)
      - must have loss + logit_scale
      - must allow count_params(eeg_model.encoder.encoder)
      - ATMS code only changes encoder_type to "EEGNex" and runs

    Notes:
      - We accept (x, subject_ids) signature but ignore subject_ids.
      - We expose `.encoder` as EEGNexEncoder, and `.encoder.encoder` is _EEGNeX.
    """
    def __init__(
        self,
        n_chans: int = 63,
        n_times: int = 250,
        embed_dim: int = 1024,
        sfreq: float | None = None,
        # Optionally pass EEGNeX hparams if you want to tune later
        **eegnex_kwargs,
    ):
        super().__init__()

        self.embed_dim = int(embed_dim)

        # backbone wrapper for count_params compatibility
        self.encoder = EEGNexEncoder(
            n_chans=n_chans,
            n_times=n_times,
            embed_dim=self.embed_dim,
            sfreq=sfreq,
            **eegnex_kwargs,
        )

        # ATMS-style CLIP loss + learnable temperature
        # Keep naming consistent with your existing training loop
        self.loss_func = ClipLoss()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, x: torch.Tensor, subject_ids=None) -> torch.Tensor:
        # Keep the structure minimal; ignore subject_ids
        feats = self.encoder(x)  # (B, 1024)

        # Optional: if your pipeline expects normalized embeddings (often in CLIP-style)
        # If ATMS pipeline already normalizes outside, you can remove this normalize.
        feats = torch.nn.functional.normalize(feats, dim=-1)

        return feats
