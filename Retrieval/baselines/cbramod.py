import torch
import torch.nn as nn
import numpy as np

from .modules.cbramodmodule import CBraMod as BD_CBraMod
from Retrieval.loss import ClipLoss


import numpy as np
import torch
import torch.nn as nn

# --------- robust imports (support different project layouts) ----------
BD_CBraMod = None
_import_errors = []

for _path in [
    "cbramodmodule",                   # same directory
    "modules.cbramodmodule",           # baselines/modules/
    "Retrieval.baselines.modules.cbramodmodule",
    "baselines.modules.cbramodmodule",
]:
    try:
        BD_CBraMod = __import__(_path, fromlist=["CBraMod"]).CBraMod
        break
    except Exception as e:
        _import_errors.append((f"from {_path} import CBraMod", repr(e)))

if BD_CBraMod is None:
    msg = "Cannot import CBraMod implementation. Tried:\n"
    msg += "\n".join([f"  - {a}: {b}" for a, b in _import_errors])
    raise ImportError(msg)

# ClipLoss import (match your ATMS script style, with fallbacks)
ClipLoss = None
_loss_import_errors = []
for _path in [
    "loss",
    "Retrieval.loss",
    "baselines.loss",
]:
    try:
        ClipLoss = __import__(_path, fromlist=["ClipLoss"]).ClipLoss
        break
    except Exception as e:
        _loss_import_errors.append((f"from {_path} import ClipLoss", repr(e)))

if ClipLoss is None:
    msg = "Cannot import ClipLoss. Tried:\n"
    msg += "\n".join([f"  - {a}: {b}" for a, b in _loss_import_errors])
    raise ImportError(msg)


class _EncoderWrapper(nn.Module):
    """
    Make .encoder.encoder exist WITHOUT circular references.
    Downstream assumes:
      - eeg_model.encoder exists
      - eeg_model.encoder.encoder exists
    """
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.encoder = backbone  # <-- backbone lives here

    def forward(self, x: torch.Tensor, *args, **kwargs):
        return self.encoder(x, *args, **kwargs)


def _pick_divisible_nhead(embed_dim: int, nhead: int) -> int:
    """
    Ensure embed_dim % nhead == 0 for MultiheadAttention.
    If not, pick a divisor of embed_dim closest to requested nhead.
    """
    embed_dim = int(embed_dim)
    nhead = int(nhead) if int(nhead) > 0 else 1
    if embed_dim % nhead == 0:
        return nhead

    divisors = []
    for d in range(1, int(embed_dim ** 0.5) + 1):
        if embed_dim % d == 0:
            divisors.append(d)
            if d != embed_dim // d:
                divisors.append(embed_dim // d)
    divisors = sorted(divisors)

    best = None
    best_key = None
    for d in divisors:
        key = (abs(d - nhead), d > nhead, d)  # distance, prefer <=nhead, then smaller
        if best_key is None or key < best_key:
            best_key = key
            best = d
    return int(best)


class CBraMod(nn.Module):
    """
    ATMS-compatible encoder_type.

    HARD constraint from your cbramodmodule.py:
      - PatchEmbedding.forward hardcodes p=101 (i.e., patch_size must be 200).
      - Therefore we must run backbone with:
            patch_size = 200
            n_times    = 200
      - Your loader provides T=250, so we crop/pad in forward to T=200.

    Alignment targets:
      1) Can be instantiated with NO args: CBraMod()
      2) Has eeg_model.encoder and eeg_model.encoder.encoder for param counting
      3) forward supports (x, subject_ids): eeg_model(eeg, subject_ids)
      4) Exposes logit_scale and loss_func used by ATMS training loop
    """
    def __init__(
        self,
        num_channels: int = 63,
        # NOTE: backbone must be built with n_times=200 due to hardcoded spectral p=101 (200//2+1)
        sequence_length: int = 200,
        out_dim: int = 1024,
        pool: str = "mean",
        # ---- CBraMod hyperparams ----
        patch_size: int = 200,       # DO NOT change unless you also change cbramodmodule.py (it hardcodes p=101)
        emb_dim: int = 200,
        dim_feedforward: int = 800,
        n_layer: int = 12,
        nhead: int = 8,
        drop_prob: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        assert pool in ("mean", "cls"), "pool must be 'mean' or 'cls'"
        self.pool = pool
        self.out_dim = out_dim

        # enforce the constraints (avoid silent mismatch)
        sequence_length = int(sequence_length)
        patch_size = int(patch_size)
        if sequence_length != 200:
            sequence_length = 200
        if patch_size != 200:
            patch_size = 200

        # ensure attention constraint
        nhead = _pick_divisible_nhead(emb_dim, nhead)

        self.expected_channels = int(num_channels)
        self.expected_T = 200

        backbone = BD_CBraMod(
            n_outputs=None,
            n_chans=self.expected_channels,
            n_times=self.expected_T,
            patch_size=patch_size,
            emb_dim=emb_dim,
            dim_feedforward=dim_feedforward,
            n_layer=n_layer,
            nhead=nhead,
            drop_prob=drop_prob,
            return_encoder_output=True,   # final_layer=Identity -> return features
            **kwargs,
        )

        # downstream expects eeg_model.encoder and eeg_model.encoder.encoder
        self.encoder = _EncoderWrapper(backbone)

        # Lazy projection to out_dim (build on first forward once we know feat dim)
        self.proj = None

        # must exist for ATMS loss and scaling
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = ClipLoss()

    def _align_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Align (B, C, T) to (B, expected_channels, expected_T) WITHOUT touching ATMS code.
        - If C != expected_channels: truncate or zero-pad channels.
        - If T != expected_T: crop or zero-pad time.
        """
        assert x.dim() == 3, f"CBraMod expects x as (B, C, T), got {tuple(x.shape)}"
        b, c, t = x.shape

        # channel align
        if c > self.expected_channels:
            x = x[:, : self.expected_channels, :]
        elif c < self.expected_channels:
            pad_c = self.expected_channels - c
            x = torch.cat([x, x.new_zeros((b, pad_c, t))], dim=1)

        # time align
        if t > self.expected_T:
            x = x[:, :, : self.expected_T]  # crop head (simple & deterministic)
        elif t < self.expected_T:
            pad_t = self.expected_T - t
            x = torch.cat([x, x.new_zeros((b, self.expected_channels, pad_t))], dim=2)

        return x

    def _build_proj_if_needed(self, feat: torch.Tensor):
        if self.proj is not None:
            return
        in_dim = feat.shape[-1]
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, self.out_dim),
        ).to(device=feat.device, dtype=feat.dtype)

    @staticmethod
    def _pool_feats(feat: torch.Tensor, pool: str) -> torch.Tensor:
        """
        feat could be:
          - (B, D)
          - (B, L, D)
          - (B, C, P, D)
        """
        if feat.dim() == 2:
            return feat
        if feat.dim() == 3:
            return feat.mean(dim=1) if pool == "mean" else feat[:, 0, :]
        if feat.dim() == 4:
            b, c, p, d = feat.shape
            tokens = feat.reshape(b, c * p, d)
            return tokens.mean(dim=1) if pool == "mean" else tokens[:, 0, :]
        b = feat.shape[0]
        tokens = feat.reshape(b, -1, feat.shape[-1])
        return tokens.mean(dim=1) if pool == "mean" else tokens[:, 0, :]

    def forward(self, x: torch.Tensor, subject_ids=None):
        """
        ATMS calls: eeg_model(eeg_data, subject_ids)
        subject_ids accepted but unused.
        x expected: (B, C, T) where T is likely 250 in your loader.
        We crop/pad to T=200 to satisfy CBraMod module constraints.
        """
        x = self._align_input(x)
        feat = self.encoder(x)          # backbone forward
        feat = self._pool_feats(feat, self.pool)
        self._build_proj_if_needed(feat)
        return self.proj(feat)
