# models/encoders/eegminer_encoder.py
"""
EEGMiner retrieval encoder wrapper.

- Output: (B, embed_dim) e.g., (B, 1024)
- No logit_scale, no loss here (kept in downstream ATMSRetrieval).
"""

from functools import partial
from typing import Tuple

import torch
from einops.layers.torch import Rearrange
from torch import nn
from torch.fft import fftfreq
import torch.nn.functional as F
import numpy as np

from braindecode.models.base import EEGModuleMixin
from Retrieval.loss import ClipLoss

_eeg_miner_methods = ["mag", "corr", "plv"]

class GeneralizedGaussianFilter(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        sequence_length,
        sample_rate,
        inverse_fourier=True,
        affine_group_delay=False,
        group_delay=(20.0,),
        f_mean=(23.0,),
        bandwidth=(44.0,),
        shape=(2.0,),
        clamp_f_mean=(1.0, 45.0),
    ):
        super(GeneralizedGaussianFilter, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.sequence_length = sequence_length
        self.sample_rate = sample_rate
        self.inverse_fourier = inverse_fourier
        self.affine_group_delay = affine_group_delay
        self.clamp_f_mean = clamp_f_mean
        if out_channels % in_channels != 0:
            raise ValueError("out_channels has to be multiple of in_channels")
        if len(f_mean) * in_channels != out_channels:
            raise ValueError("len(f_mean) * in_channels must equal out_channels")
        if len(bandwidth) * in_channels != out_channels:
            raise ValueError("len(bandwidth) * in_channels must equal out_channels")
        if len(shape) * in_channels != out_channels:
            raise ValueError("len(shape) * in_channels must equal out_channels")

        # Range from 0 to half sample rate, normalized
        self.n_range = nn.Parameter(
            torch.tensor(
                list(
                    fftfreq(n=sequence_length, d=1 / sample_rate)[
                        : sequence_length // 2
                    ]
                )
                + [sample_rate / 2]
            )
            / (sample_rate / 2),
            requires_grad=False,
        )

        # Trainable filter parameters
        self.f_mean = nn.Parameter(
            torch.tensor(f_mean * in_channels) / (sample_rate / 2), requires_grad=True
        )
        self.bandwidth = nn.Parameter(
            torch.tensor(bandwidth * in_channels) / (sample_rate / 2),
            requires_grad=True,
        )  # full width half maximum
        self.shape = nn.Parameter(torch.tensor(shape * in_channels), requires_grad=True)

        # Normalize group delay so that group_delay=1 corresponds to 1000ms
        self.group_delay = nn.Parameter(
            torch.tensor(group_delay * in_channels) / 1000,
            requires_grad=affine_group_delay,
        )

        # Construct filters from parameters and register as a buffer so
        # torch.export / tracing treats it as a proper module buffer
        # (it will be recomputed in forward anyway).
        self.register_buffer("filters", self.construct_filters(), persistent=False)

    @staticmethod
    def exponential_power(x, mean, fwhm, shape):
        mean = mean.unsqueeze(1)
        fwhm = fwhm.unsqueeze(1)
        shape = shape.unsqueeze(1)
        log2 = torch.log(torch.tensor(2.0, device=x.device, dtype=x.dtype))
        scale = fwhm / (2 * log2 ** (1 / shape))
        # Add small constant to difference between x and mean since grad of 0 ** shape is nan
        return torch.exp(-((((x - mean).abs() + 1e-8) / scale) ** shape))

    def construct_filters(self):
        # Clamp parameters
        self.f_mean.data = torch.clamp(
            self.f_mean.data,
            min=self.clamp_f_mean[0] / (self.sample_rate / 2),
            max=self.clamp_f_mean[1] / (self.sample_rate / 2),
        )
        self.bandwidth.data = torch.clamp(
            self.bandwidth.data, min=1.0 / (self.sample_rate / 2), max=1.0
        )
        self.shape.data = torch.clamp(self.shape.data, min=2.0, max=3.0)

        # Create magnitude response with gain=1 -> (channels, freqs)
        mag_response = self.exponential_power(
            self.n_range, self.f_mean, self.bandwidth, self.shape * 8 - 14
        )
        mag_response = mag_response / mag_response.max(dim=-1, keepdim=True)[0]

        # Create phase response, scaled so that normalized group_delay=1
        # corresponds to group delay of 1000ms.
        phase = torch.linspace(
            0,
            self.sample_rate,
            self.sequence_length // 2 + 1,
            device=mag_response.device,
            dtype=mag_response.dtype,
        )
        phase = phase.expand(mag_response.shape[0], -1)  # repeat for filter channels
        pha_response = -self.group_delay.unsqueeze(-1) * phase * torch.pi

        # Create real and imaginary parts of the filters
        real = mag_response * torch.cos(pha_response)
        imag = mag_response * torch.sin(pha_response)

        # Stack real and imaginary parts to create filters
        # -> (channels, freqs, 2)
        filters = torch.stack((real, imag), dim=-1)

        return filters

    def forward(self, x):
        """
        Applies the generalized Gaussian filters to the input signal.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape `(..., in_channels, sequence_length)`.

        Returns
        -------
        torch.Tensor
            The filtered signal. If `inverse_fourier` is True, returns the signal in the time domain
            with shape `(..., out_channels, sequence_length)`. Otherwise, returns the signal in the
            frequency domain with shape `(..., out_channels, freq_bins, 2)`.

        """
        # Construct filters from parameters
        self.filters = self.construct_filters()
        # Preserving the original dtype.
        dtype = x.dtype
        # Apply FFT -> (..., channels, freqs, 2)
        x = torch.fft.rfft(x, dim=-1)
        x = torch.view_as_real(x)  # separate real and imag

        # Repeat channels in case of multiple filters per channel
        x = torch.repeat_interleave(x, self.out_channels // self.in_channels, dim=-3)

        # Apply filters in the frequency domain
        x = x * self.filters

        # Apply inverse FFT if requested
        if self.inverse_fourier:
            x = torch.view_as_complex(x)
            x = torch.fft.irfft(x, n=self.sequence_length, dim=-1)

        x = x.to(dtype)

        return x


def hilbert_freq(x, forward_fourier=True):
    if forward_fourier:
        x = torch.fft.rfft(x, norm=None, dim=-1)
        x = torch.view_as_real(x)
    x = x * 2.0
    x[..., 0, :] = x[..., 0, :] / 2.0  # Don't multiply the DC-term by 2
    x = F.pad(
        x, [0, 0, 0, x.shape[-2] - 2]
    )  # Fill Fourier coefficients to retain shape
    x = torch.view_as_complex(x)
    x = torch.fft.ifft(x, norm=None, dim=-1)  # returns complex signal
    x = torch.view_as_real(x)

    return x


def plv_time(x, forward_fourier=True, epsilon: float = 1e-6):
    # Compute the analytic signal using the Hilbert transform.
    # x_a has separate real and imaginary parts.
    analytic_signal = hilbert_freq(x, forward_fourier)
    # Calculate the amplitude (magnitude) of the analytic signal.
    # Adding a small epsilon (1e-6) to avoid division by zero.
    amplitude = torch.sqrt(
        analytic_signal[..., 0] ** 2 + analytic_signal[..., 1] ** 2 + 1e-6
    )
    # Normalize the analytic signal to obtain unit vectors (phasors).
    unit_phasor = analytic_signal / amplitude.unsqueeze(-1)

    # Compute the real part of the outer product between phasors of
    # different channels.
    real_real = torch.matmul(unit_phasor[..., 0], unit_phasor[..., 0].transpose(-2, -1))

    # Compute the imaginary part of the outer product between phasors of
    # different channels.
    imag_imag = torch.matmul(unit_phasor[..., 1], unit_phasor[..., 1].transpose(-2, -1))

    # Compute the cross-terms for the real and imaginary parts.
    real_imag = torch.matmul(unit_phasor[..., 0], unit_phasor[..., 1].transpose(-2, -1))
    imag_real = torch.matmul(unit_phasor[..., 1], unit_phasor[..., 0].transpose(-2, -1))

    # Combine the real and imaginary parts to form the complex correlation.
    correlation_real = real_real + imag_imag
    correlation_imag = real_imag - imag_real

    # Determine the number of time points (or frequency bins if in Fourier domain).
    time = amplitude.shape[-1]

    # Calculate the PLV by averaging the magnitude of the complex correlation over time.
    # epsilon is small numerical value to ensure positivity constraint on the complex part
    plv_matrix = (
        1 / time * torch.sqrt(correlation_real**2 + correlation_imag**2 + epsilon)
    )

    return plv_matrix


class _EEGMiner(EEGModuleMixin, nn.Module):
    """
    Minimal EEGMiner core (same logic as you pasted):
    input:  (B, C, T)
    output: (B, n_outputs)
    """

    def __init__(
        self,
        method: str = "plv",
        n_chans=None,
        n_outputs=None,
        n_times=None,
        chs_info=None,
        input_window_seconds=None,
        sfreq=None,
        filter_f_mean=(23.0, 23.0),
        filter_bandwidth=(44.0, 44.0),
        filter_shape=(2.0, 2.0),
        group_delay=(20.0, 20.0),
        clamp_f_mean=(1.0, 45.0),
    ):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq

        self.filter_f_mean = filter_f_mean
        self.filter_bandwidth = filter_bandwidth
        self.filter_shape = filter_shape
        self.n_filters = len(self.filter_f_mean)
        self.group_delay = group_delay
        self.clamp_f_mean = clamp_f_mean
        self.method = method.lower()

        if self.method not in _eeg_miner_methods:
            raise ValueError(f"Invalid method {self.method}, choose from {_eeg_miner_methods}")

        if self.method in ["mag", "corr"]:
            inverse_fourier = True
            in_channels = self.n_chans
            out_channels = self.n_chans * self.n_filters
        else:
            inverse_fourier = False
            in_channels = 1
            out_channels = 1 * self.n_filters

        self.filter = GeneralizedGaussianFilter(
            in_channels=in_channels,
            out_channels=out_channels,
            sequence_length=self.n_times,
            sample_rate=self.sfreq,
            f_mean=self.filter_f_mean,
            bandwidth=self.filter_bandwidth,
            shape=self.filter_shape,
            affine_group_delay=False,
            inverse_fourier=inverse_fourier,
            group_delay=self.group_delay,
            clamp_f_mean=self.clamp_f_mean,
        )

        if self.method == "mag":
            self.method_forward = self._apply_mag_forward
            self.n_features = self.n_chans * self.n_filters
            self.ensure_dim = nn.Identity()
        elif self.method == "corr":
            self.method_forward = partial(
                self._apply_corr_forward,
                n_chans=self.n_chans,
                n_filters=self.n_filters,
                n_times=self.n_times,
            )
            self.n_features = self.n_filters * self.n_chans * (self.n_chans - 1) // 2
            self.ensure_dim = nn.Identity()
        else:  # plv
            # PLV path expects (..., 1, T) before filter
            self.method_forward = partial(self._apply_plv, n_chans=self.n_chans)
            self.ensure_dim = lambda x: x.unsqueeze(-2)  # (B,C,T)->(B,C,1,T)
            self.n_features = (self.n_filters * self.n_chans * (self.n_chans - 1)) // 2

        self.batch_layer = nn.BatchNorm1d(self.n_features, affine=False)
        self.final_layer = nn.Linear(self.n_features, self.n_outputs)
        nn.init.zeros_(self.final_layer.bias)

    def forward(self, x):
        batch = x.shape[0]
        x = self.ensure_dim(x)
        x = self.filter(x)
        x = self.method_forward(x=x, batch=batch)
        x = x.reshape(batch, self.n_features)
        x = self.batch_layer(x)
        x = self.final_layer(x)
        return x

    @staticmethod
    def _apply_mag_forward(x, batch=None):
        x = x * x
        x = x.mean(dim=-1)
        x = torch.sqrt(x)
        return x

    @staticmethod
    def _apply_corr_forward(x, batch, n_chans, n_filters, n_times, epilson: float = 1e-6):
        x = x.reshape(batch, n_chans, n_filters, n_times).transpose(-3, -2)
        x = (x - x.mean(dim=-1, keepdim=True)) / torch.sqrt(x.var(dim=-1, keepdim=True) + epilson)
        x = torch.matmul(x, x.transpose(-2, -1)) / x.shape[-1]
        x = x.permute(0, 2, 3, 1)  # [B, chans, chans, n_filters]
        x = x.abs()
        triu = torch.triu_indices(n_chans, n_chans, 1)
        x = x[:, triu[0], triu[1], :]
        return x

    @staticmethod
    def _apply_plv(x, n_chans, batch=None):
        # x: (B, C, F, T) or similar after filter; keep same logic as your snippet
        x = x.transpose(-4, -3)  # swap electrodes and filters
        x = plv_time(x, forward_fourier=False)
        x = x.permute(0, 2, 3, 1)  # [B, chans, chans, n_filters]
        triu = torch.triu_indices(n_chans, n_chans, 1)
        x = x[:, triu[0], triu[1], :]
        return x


class EEGMinerEncoderCompat(nn.Module):
    """
    IMPORTANT: this wrapper exists solely to satisfy your downstream parameter counting:

      encoder_params  = count_params(eeg_model.encoder)
      backbone_params = count_params(eeg_model.encoder.encoder)

    So:
      - this module is assigned to eeg_model.encoder
      - it contains self.encoder (the actual backbone)

    forward signature is made compatible with iTransformer usage:
      forward(x, x_mark_enc=None, subject_ids=None)
    """

    def __init__(
        self,
        n_chans: int = 63,
        n_times: int = 250,
        sfreq: float = 250.0,
        embed_dim: int = 1024,
        method: str = "mag",
        l2norm: bool = True,
        # filter params
        filter_f_mean=(23.0, 23.0),
        filter_bandwidth=(44.0, 44.0),
        filter_shape=(2.0, 2.0),
        group_delay=(20.0, 20.0),
        clamp_f_mean=(1.0, 45.0),
    ):
        super().__init__()
        self.l2norm = l2norm

        # This is the "backbone" used for backbone_params counting.
        self.encoder = _EEGMiner(
            method=method,
            n_chans=n_chans,
            n_outputs=embed_dim,
            n_times=n_times,
            sfreq=sfreq,
            filter_f_mean=filter_f_mean,
            filter_bandwidth=filter_bandwidth,
            filter_shape=filter_shape,
            group_delay=group_delay,
            clamp_f_mean=clamp_f_mean,
        )

    def forward(self, x, x_mark_enc=None, subject_ids=None):
        z = self.encoder(x)  # (B, embed_dim)
        if self.l2norm:
            z = F.normalize(z, dim=-1)
        return z


class EEGMiner(nn.Module):
    """
    Drop-in replacement for ATMS in ATMS_retrieval_metrics.py.

    Must satisfy:
      - eeg_model.encoder exists
      - eeg_model.encoder.encoder exists
      - eeg_model.logit_scale exists
      - eeg_model.loss_func exists and callable like loss_func(eeg_feat, img_feat, logit_scale)
      - forward(eeg, subject_ids) -> (B, 1024)
    """

    def __init__(
        self,
        num_channels: int = 63,
        sequence_length: int = 250,
        sfreq: float = 250.0,
        embed_dim: int = 1024,
        method: str = "mag",
    ):
        super().__init__()

        # ---- the encoder attribute REQUIRED by your downstream ----
        self.encoder = EEGMinerEncoderCompat(
            n_chans=num_channels,
            n_times=sequence_length,
            sfreq=sfreq,
            embed_dim=embed_dim,
            method=method,
            l2norm=True,
        )

        # ---- REQUIRED by your downstream training code ----
        # NOTE: your script multiplies logit_scale directly (no exp), and init uses log(1/0.07).
        # Keep identical to your ATMS class for behavior parity.
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = ClipLoss()

    def forward(self, x, subject_ids=None):
        # subject_ids ignored (EEGMiner has no subject embedding); keep signature compatible
        z = self.encoder(x, None, subject_ids)  # (B, 1024)
        return z
