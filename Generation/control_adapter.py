# control_adapter.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


def _zero_module(m: nn.Module) -> nn.Module:
    """Zero-init a module's parameters (ControlNet-style safe start)."""
    for p in m.parameters():
        nn.init.zeros_(p)
    return m


class ControlAdapterXS(nn.Module):
    """
    A lightweight ControlNet-XS-like adapter:
      control_image (B,3,H,W) -> a list of residuals aligned to UNet down blocks + one mid residual.
    The backbone UNet is frozen; only this adapter is trained.

    This adapter is architecture-aware via unet.config.block_out_channels.
    """

    def __init__(
        self,
        block_out_channels: Tuple[int, ...],
        in_channels: int = 3,
        base_channels: int = 64,
    ):
        super().__init__()
        self.block_out_channels = tuple(block_out_channels)

        # Stem: map RGB control to a small feature
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.SiLU(),
        )

        # For each UNet down block, build a tiny conv tower that:
        # - downsamples (stride=2) when moving to deeper scales
        # - projects to the block's channel size
        self.down_projs = nn.ModuleList()
        in_ch = base_channels
        for i, out_ch in enumerate(self.block_out_channels):
            # For i=0, usually no downsample needed (match highest res)
            # For i>0, we downsample once per level (simple, robust)
            layers = []
            if i > 0:
                layers.append(nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1))
                layers.append(nn.SiLU())

            # light mixing
            layers.append(nn.Conv2d(in_ch, in_ch, 3, padding=1))
            layers.append(nn.SiLU())

            # project to UNet block channels; last conv is zero-init for safe injection
            layers.append(_zero_module(nn.Conv2d(in_ch, out_ch, 1)))

            self.down_projs.append(nn.Sequential(*layers))
            # in_ch = in_ch  # keep base width small
            in_ch = out_ch

        # Mid residual: project deepest feature to last block channels (also zero-init)
        self.mid_proj = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.SiLU(),
            _zero_module(nn.Conv2d(in_ch, self.block_out_channels[-1], 1)),
        )

    def forward(
        self,
        control_image: torch.Tensor,
        target_spatial: Optional[List[Tuple[int, int]]] = None,
    ):
        """
        Returns:
          down_residuals: list[Tensor] length = len(block_out_channels)
          mid_residual: Tensor
        Notes:
          - We optionally interpolate to match exact UNet shapes (recommended).
        """
        x = self.stem(control_image)

        down_residuals = []
        feat = x
        for i, proj in enumerate(self.down_projs):
            feat = proj(feat)

            if target_spatial is not None:
                th, tw = target_spatial[i]
                if feat.shape[-2:] != (th, tw):
                    feat = F.interpolate(feat, size=(th, tw), mode="bilinear", align_corners=False)

            down_residuals.append(feat)

        mid = self.mid_proj(feat)
        return down_residuals, mid


def enable_only_control_adapter_trainable(pipe, control_adapter: nn.Module):
    """
    Freeze everything in pipe (UNet/VAE/text encoders/IP-Adapter etc.) and unfreeze only control_adapter.
    Returns a list of trainable parameter names for sanity check.
    """
    # Freeze pipeline modules commonly present in SDXL pipelines
    for name in ["unet", "vae", "text_encoder", "text_encoder_2", "image_encoder"]:
        if hasattr(pipe, name) and getattr(pipe, name) is not None:
            getattr(pipe, name).requires_grad_(False)

    # Freeze attention processors / IP-Adapter weights if they exist inside UNet
    if hasattr(pipe, "unet") and pipe.unet is not None:
        pipe.unet.requires_grad_(False)

    # Unfreeze only control adapter
    control_adapter.requires_grad_(True)

    trainable = []
    for n, p in control_adapter.named_parameters():
        if p.requires_grad:
            trainable.append(f"control_adapter.{n}")
    return trainable
