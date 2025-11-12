import torch
import torch.nn as nn
from torchaudio.transforms import Spectrogram, InverseSpectrogram


class InvSpecBlock(nn.Module):
    def __init__(
        self, config, device="cpu"
    ):
        super().__init__()

        n_fft_recon, \
            hop_length_recon, \
            win_length_recon = config["model"]["blocks"][0]["params"].values()

        self.device = torch.device(device)

        # ---- Inverse spectrogram ----
        self.reconspec_to_waveform = InverseSpectrogram(
            n_fft=n_fft_recon,
            hop_length=hop_length_recon,
            win_length=win_length_recon,
            center=True
        )
        # Ensure window on correct device
        self._move_spec_windows(self.device)

    def _move_spec_windows(self, device):
        if hasattr(self.reconspec_to_waveform, "window") and self.reconspec_to_waveform.window is not None:
            self.reconspec_to_waveform.window = self.reconspec_to_waveform.window.to(
                device)

    def to(self, device):
        """Override nn.Module.to() to propagate device to InverseSpectrogram window."""
        device = torch.device(device)
        super().to(device)
        self._move_spec_windows(device)
        self.device = device
        return self

    def forward(self, x):
        # x: (B, 2, 1001, 401)
        # Ensure window device matches input
        if x.device != self.device:
            self._move_spec_windows(x.device)
            self.device = x.device

        # ---- Reconstructed waveform ----
        recon_wave = self.reconspec_to_waveform(
            x[:, 0] + 1j * x[:, 1])  # (B, T)
        # (B,1,T) to match input channel dim
        recon_wave = recon_wave.unsqueeze(1)

        return recon_wave
