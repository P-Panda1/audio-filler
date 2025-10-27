import torch
import torch.nn as nn
from torchaudio.transforms import Spectrogram, InverseSpectrogram


class InvSpecBlock(nn.Module):
    def __init__(
        self,
        n_fft_recon,
        hop_length_recon,
        win_length_recon
    ):
        super().__init__()

        # ---- Inverse spectrogram ----
        self.reconspec_to_waveform = InverseSpectrogram(
            n_fft=n_fft_recon,
            hop_length=hop_length_recon,
            win_length=win_length_recon,
            center=True
        )

    def forward(self, x):
        # x: (B, 2, 1001, 401)

        # ---- Reconstructed waveform ----
        recon_wave = self.reconspec_to_waveform(
            x[:, 0] + 1j * x[:, 1])  # (B, T)
        # (B,1,T) to match input channel dim
        recon_wave = recon_wave.unsqueeze(1)

        return recon_wave
