import torch
import torch.nn as nn
from torchaudio.transforms import Spectrogram, InverseSpectrogram


class SpectrogramBlock(nn.Module):
    def __init__(
        self,
        n_fft_f,
        hop_length_f,
        window_size_f,
        n_fft_t,
        hop_length_t,
        window_size_t,
        n_fft_recon,
        hop_length_recon,
        win_length_recon
    ):
        super().__init__()

        # ---- Frequency spectrogram ----
        self.to_spec_f = Spectrogram(
            n_fft=n_fft_f,
            hop_length=hop_length_f,
            win_length=window_size_f,
            power=None,
            center=True
        )

        # ---- Time spectrogram ----
        self.to_spec_t = Spectrogram(
            n_fft=n_fft_t,
            hop_length=hop_length_t,
            win_length=window_size_t,
            power=None,
            center=True
        )

        # ---- Reconstruction spectrogram ----
        self.recon_to_spec = Spectrogram(
            n_fft=n_fft_recon,
            hop_length=hop_length_recon,
            win_length=win_length_recon,
            power=None,
            center=True
        )

        # ---- Inverse spectrogram ----
        self.reconspec_to_waveform = InverseSpectrogram(
            n_fft=n_fft_recon,
            hop_length=hop_length_recon,
            win_length=win_length_recon,
            center=True
        )

    def forward(self, x):
        # x: (B, 1, T)
        spec_f = self.to_spec_f(x)  # (B, 1, F_f, T_f)
        spec_f = torch.stack([spec_f.real, spec_f.imag],
                             dim=1)  # (B, 2, F_f, T_f)
        spec_t = self.to_spec_t(x)  # (B, 1, F_t, T_t)
        spec_t = torch.stack([spec_t.real, spec_t.imag],
                             dim=1)  # (B, 2, F_t, T_t)
        spec_r = self.recon_to_spec(x)  # (B, 1, F_r, T_r)
        spec_r = torch.stack([spec_r.real, spec_r.imag],
                             dim=1)  # (B, 2, F_r, T_r)
        recon_wave = self.reconspec_to_waveform(
            spec_r[:, 0, :, :] + 1j * spec_r[:, 1, :, :])  # (B, 1, T)
        return {
            "freq_spec": spec_f,
            "time_spec": spec_t,
            "recon_spec": spec_r,
            "recon_wave": recon_wave
        }
