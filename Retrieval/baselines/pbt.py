from .modules.pbtmodule import PBT as _PBT
import torch
import torch.nn as nn
from Retrieval.loss import ClipLoss
import numpy as np


class PBTBackbone(nn.Module):
    """
    Wrapper around PBT.

    Input : (B, 63, 250)
    Output: (B, 1024)   # CLS token readout through final_layer
    Exposes `.encoder` so count_params(eeg_model.encoder.encoder) works.
    """
    def __init__(
        self,
        n_chans: int = 63,
        n_times: int = 250,
        d_input: int = 50,
        embed_dim: int = 250,
        num_layers: int = 4,
        num_heads: int = 5,
        drop_prob: float = 0.1,
        learnable_cls: bool = True,
        bias_transformer: bool = False,
        activation: type[nn.Module] = nn.GELU,
    ):
        super().__init__()
        assert n_times % d_input == 0
        assert embed_dim % num_heads == 0

        self.pbt = _PBT(
            n_chans=n_chans,
            n_times=n_times,
            n_outputs=1024,   # final_layer output dim
            d_input=d_input,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            drop_prob=drop_prob,
            learnable_cls=learnable_cls,
            bias_transformer=bias_transformer,
            activation=activation,
        )

        # for parameter counting: eeg_model.encoder.encoder
        self.encoder = self.pbt.transformer_encoder

    def forward(self, X: torch.Tensor, x_mark_enc=None, subject_ids=None) -> torch.Tensor:
        # keep original forward structure as much as possible
        X, int_pos = self.pbt.patch_signal(X)

        tokens = self.pbt.patching_projection(X)

        cls_token = self.pbt.cls_token.expand(X.size(0), 1, -1)
        cls_idx = torch.zeros((X.size(0), 1), dtype=torch.long, device=X.device)  # unused, keep for fidelity

        tokens = torch.cat([cls_token, tokens], dim=1)
        pos_emb = self.pbt.pos_embedding(int_pos)
        transformer_out = self.pbt.transformer_encoder(tokens + pos_emb)

        return self.pbt.final_layer(transformer_out[:, 0])  # <-- FIX


class PBT(nn.Module):
    """
    Drop-in replacement for ATMS in retrieval script.
    forward(eeg_data, subject_ids) -> (B, 1024)
    Provides: loss_func, logit_scale.
    Provides: encoder and encoder.encoder (for count parameters).
    """
    def __init__(self, num_channels=63, sequence_length=250, num_subjects=2, num_features=64, num_latents=1024, num_blocks=1):
        super().__init__()

        self.encoder = PBTBackbone(
            n_chans=num_channels,
            n_times=sequence_length,
            d_input=25,
            embed_dim=sequence_length,  # 250
            num_layers=4,
            num_heads=5,
            drop_prob=0.1,
            learnable_cls=True,
            bias_transformer=False,
            activation=nn.GELU,
        )

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = ClipLoss()

    def forward(self, x, subject_ids):
        return self.encoder(x, None, subject_ids)  # (B,1024)
