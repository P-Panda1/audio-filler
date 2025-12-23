import torch
import torch.nn as nn
from src.blocks.SpectrogramBlock import SpectrogramBlock
from src.blocks.InvSpecBlock import InvSpecBlock
import src.models.encoders.encoder_model_1 as encoder_module
import src.models.decoders.decoder_model_1 as decoder_module


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


device = get_device()


class EncoderDecoderModel(nn.Module):
    def __init__(self, configs, latent_dim=1024, class_size=15, device="cpu", mode="default"):
        super().__init__()
        encoder_config, \
            decoder_config, \
            spectrogram_config, \
            inv_spectrogram_config = configs
        self.spectrogram = SpectrogramBlock(spectrogram_config, device)
        self.inv_spectrogram = InvSpecBlock(
            inv_spectrogram_config, device=device)
        self.encoder = encoder_module.EncoderModel1(encoder_config, latent_dim)
        self.decoder = decoder_module.DecoderModel1(
            decoder_config, latent_dim, class_size)
        # default mode accepts waveform, train mode accepts dict with the spectogram dictionary already given
        self.mode = mode
        self.scale_mlp = nn.Sequential(
            nn.Linear(2, 1),  # input: [freq_max, time_max] -> output: [scale]
        )

    def to(self, device):
        super().to(device)
        # propagate device to spectrogram-related modules
        if hasattr(self.spectrogram, "to"):
            self.spectrogram.to(device)
        if hasattr(self.inv_spectrogram, "to"):
            self.inv_spectrogram.to(device)
        return self

    def forward(self, x):
        if self.mode == "default":
            spec_dict = self.spectrogram(x)
        elif self.mode == "train":
            spec_dict = x
        latent, mu, var, freq_max, time_max = self.encode(x)
        # We depricate class_out entirely
        recon = self.decode(latent, freq_max, time_max)

        return recon, mu, var, spec_dict['recon_spec']

    def encode(self, x):
        x = x.to(device)

        if self.mode == "default":
            spec_dict = self.spectrogram(x)
        elif self.mode == "train":
            spec_dict = x

        freq = spec_dict['freq_spec']
        time = spec_dict['time_spec']

        # For data parallelisation
        if freq.dim() == 5:
            freq = freq.squeeze(2)
        if time.dim() == 5:
            time = time.squeeze(2)

        # --- Per-channel normalization ---
        # Compute max per sample & channel
        freq_max = freq.abs().amax(
            dim=[2, 3], keepdim=True)  # shape [B, C, 1, 1]
        time_max = time.abs().amax(
            dim=[2, 3], keepdim=True)  # shape [B, C, 1, 1]

        freq_norm = freq / (freq_max + 1e-8)
        time_norm = time / (time_max + 1e-8)

        mu, var = self.encoder(freq_norm, time_norm)
        latent = self.encoder.reparameterize(mu, var)
        return latent, mu, var, freq_max, time_max

    def decode(self, latent, freq_max=None, time_max=None):
        recon = self.decoder(latent)
        if freq_max is not None and time_max is not None:
            # flatten per-channel max values
            B, C, _, _ = recon.shape
            # shape [B*C, 2] for feeding into MLP
            scale_input = torch.stack([freq_max.squeeze(-1).squeeze(-1),
                                       time_max.squeeze(-1).squeeze(-1)], dim=-1).view(-1, 2)
            scale = self.scale_mlp(scale_input)  # shape [B*C, 1]
            scale = scale.view(B, C, 1, 1)
            recon = recon * scale
        return recon

    def generate(self, latent, freq_max=None, time_max=None):
        recon, _ = self.decoder(latent, freq_max, time_max)
        recon_complex = torch.complex(recon[:, 0, :, :], recon[:, 1, :, :])
        audio_recon = self.inv_spectrogram(recon_complex)
        return audio_recon
