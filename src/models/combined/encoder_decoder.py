import torch
import torch.nn as nn
from src.blocks.SpectrogramBlock import SpectrogramBlock
from src.blocks.InvSpecBlock import InvSpecBlock
import src.models.encoders.encoder_model_1 as encoder_module
import src.models.decoders.decoder_model_1 as decoder_module


class EncoderDecoderModel(nn.Module):
    def __init__(self, configs, latent_dim=512, class_size=15):
        super().__init__()
        encoder_config, \
            decoder_config, \
            spectrogram_config, \
            inv_spectrogram_config = configs
        self.spectrogram = SpectrogramBlock(spectrogram_config)
        self.inv_spectrogram = InvSpecBlock(inv_spectrogram_config)
        self.encoder = encoder_module.EncoderModel1(encoder_config, latent_dim)
        self.decoder = decoder_module.DecoderModel1(
            decoder_config, latent_dim, class_size)

    def forward(self, x):
        freq, time = self.spectrogram(x)
        mu, var = self.encoder(freq, time)
        latent = self.encoder.reparameterize(mu, var)
        recon, class_out = self.decoder(latent)
        return recon, class_out, mu, var

    def encode(self, x):
        freq, time = self.spectrogram(x)
        mu, var = self.encoder(freq, time)
        latent = self.encoder.reparameterize(mu, var)
        return latent, mu, var

    def generate(self, latent):
        recon, _ = self.decoder(latent)
        recon_complex = torch.complex(recon[:, 0, :, :], recon[:, 1, :, :])
        audio_recon = self.inv_spectrogram(recon_complex)

        return audio_recon
