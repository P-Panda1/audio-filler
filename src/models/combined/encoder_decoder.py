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
    def __init__(self, configs, latent_dim=1024, class_size=15, device="cpu"):
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

    def to(self, device):
        super().to(device)
        # propagate device to spectrogram-related modules
        if hasattr(self.spectrogram, "to"):
            self.spectrogram.to(device)
        if hasattr(self.inv_spectrogram, "to"):
            self.inv_spectrogram.to(device)
        return self

    def forward(self, x):
        spec_dict = self.spectrogram(x)
        latent, mu, var = self.encode(x)
        recon, _ = self.decode(latent)  # We depricate class_out entirely

        return recon, mu, var, spec_dict['recon_spec']

    def encode(self, x):
        x = x.to(device)
        spec_dict = self.spectrogram(x)
        freq = spec_dict['freq_spec']
        time = spec_dict['time_spec']
        mu, var = self.encoder(freq, time)
        latent = self.encoder.reparameterize(mu, var)
        return latent, mu, var

    def decode(self, latent):
        recon, _ = self.decoder(latent)
        return recon

    def generate(self, latent):
        recon, _ = self.decoder(latent)
        recon_complex = torch.complex(recon[:, 0, :, :], recon[:, 1, :, :])
        audio_recon = self.inv_spectrogram(recon_complex)
        return audio_recon
