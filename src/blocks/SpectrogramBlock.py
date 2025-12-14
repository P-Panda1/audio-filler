import torch
import torch.nn as nn
from torchaudio.transforms import Spectrogram, InverseSpectrogram


class SpectrogramBlock(nn.Module):
    def __init__(
        self,
        config,
        device="cpu"
    ):
        super().__init__()
        n_fft_f, \
            hop_length_f, \
            window_size_f, \
            n_fft_t, \
            hop_length_t, \
            window_size_t, \
            n_fft_recon, \
            hop_length_recon, \
            win_length_recon = config["model"]["blocks"][0]["params"].values()

        self.device = torch.device(device)
        # ---- Frequency spectrogram ----
        self.to_spec_f = Spectrogram(
            n_fft=n_fft_f,
            hop_length=hop_length_f,
            win_length=window_size_f,
            power=None,
            center=True,
        )

        # ---- Time spectrogram ----
        self.to_spec_t = Spectrogram(
            n_fft=n_fft_t,
            hop_length=hop_length_t,
            win_length=window_size_t,
            power=None,
            center=True,
        )

        # ---- Reconstruction spectrogram ----
        self.recon_to_spec = Spectrogram(
            n_fft=n_fft_recon,
            hop_length=hop_length_recon,
            win_length=win_length_recon,
            power=None,
            center=True,
        )
        # Move internal windows to the correct device if they exist
        self._move_spec_windows(self.device)

    def _move_spec_windows(self, device):
        """Ensures internal window tensors are on the same device."""
        for spec in [self.to_spec_f, self.to_spec_t, self.recon_to_spec]:
            if hasattr(spec, "window") and spec.window is not None:
                spec.window = spec.window.to(device)

    def to(self, device):
        """Override nn.Module.to() to propagate device to internal Spectrograms."""
        device = torch.device(device)
        super().to(device)
        self._move_spec_windows(device)
        self.device = device
        return self

    @staticmethod
    def _complex_to_logmag(spec):
        # spec: (B, 1, F, T) complex tensor
        spec = spec.squeeze(1)  # remove singleton channel dim → (B, F, T)
        magnitude = torch.abs(spec)
        logmag = 20 * torch.log10(magnitude + 1e-10)
        # stack along new channel dim: (B, 3, F, T)
        return torch.stack([spec.real, spec.imag, logmag], dim=1)

    def forward(self, x):
        # x: (B, 1, T)

        # Move spectrogram windows dynamically if input device changed
        if x.device != self.device:
            self._move_spec_windows(x.device)
            self.device = x.device

        # FIX: Disable mixed precision (FP16) for this block only.
        # This allows cuFFT to handle non-power-of-2 sizes (like 4000) without crashing.
        with torch.amp.autocast('cuda', enabled=False):
            # 1. Force input to Float32
            x = x.float()

            # 2. Run STFT and LogMag in safe Float32
            spec_f = self._complex_to_logmag(self.to_spec_f(x))
            spec_t = self._complex_to_logmag(self.to_spec_t(x))
            spec_r = self._complex_to_logmag(self.recon_to_spec(x))

        return {
            "freq_spec": spec_f,
            "time_spec": spec_t,
            "recon_spec": spec_r
        }
