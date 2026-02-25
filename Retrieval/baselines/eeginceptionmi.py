from .modules.eeginception_mi import EEGInceptionMI as _EEGInceptionMI
import torch
import torch.nn as nn
from Retrieval.loss import ClipLoss
import numpy as np


# ========================= EEGInceptionMI wrapper (ATMS-compatible) =========================
class IMIEncoder(nn.Module):
    """
    Thin wrapper so that:
      - `eeg_model.encoder` exists (for count_params)
      - `eeg_model.encoder.encoder` exists (the actual braindecode backbone)
      - forward(x, subject_ids) works (subject_ids ignored)
      - output is (B, 1024) to match ATMS downstream retrieval logic.
    """
    def __init__(self, n_chans: int = 63, n_times: int = 250, sfreq: int = 250, out_dim: int = 1024):
        super().__init__()
        if _EEGInceptionMI is None:
            raise ImportError(
                "braindecode.models.EEGInceptionMI is not available in your environment. "
                "Please upgrade braindecode or install a version that includes EEGInceptionMI."
            )
        # NOTE: We set n_outputs=out_dim so the model returns an embedding vector rather than class logits.
        # This keeps the rest of the retrieval pipeline unchanged (expects B x 1024 features).
        self.encoder = _EEGInceptionMI(
            n_chans=n_chans,
            n_outputs=out_dim,
            n_times=n_times,
            sfreq=sfreq,
        )

    def forward(self, x: torch.Tensor, subject_ids=None) -> torch.Tensor:
        # subject_ids is intentionally ignored (EEGInceptionMI is not subject-conditioned).
        return self.encoder(x)


class EEGInceptionMI(nn.Module):
    """ATMS-compatible model that swaps ATMS backbone -> EEGInceptionMI, keeping loss/logit_scale interface."""
    def __init__(self, n_chans: int = 63, n_times: int = 250, sfreq: int = 250, proj_dim: int = 1024):
        super().__init__()
        # Expose encoder + encoder.encoder for the existing parameter counting code:
        #   encoder_params   = count_params(eeg_model.encoder)
        #   backbone_params  = count_params(eeg_model.encoder.encoder)
        self.encoder = IMIEncoder(n_chans=n_chans, n_times=n_times, sfreq=sfreq, out_dim=proj_dim)

        # Keep ATMS retrieval training API unchanged
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = ClipLoss()

    def forward(self, x: torch.Tensor, subject_ids=None) -> torch.Tensor:
        # Return B x 1024 features (same shape as ATMS.proj_eeg output).
        out = self.encoder(x, subject_ids)
        return out
# ============================================================================================
