import torch
import torch.nn as nn
from src.blocks.ConvBlock import ConvBlock
from src.blocks.SpectogramBlock import SpectrogramBlock
from src.blocks.InvSpecBlock import InvSpecBlock
import yaml

# Load YAML
with open("../../configs/spectogram.yaml", "r") as f:
    spec_cfg = yaml.safe_load(f)
    spec_params = spec_cfg["model"]["blocks"][0]["params"]

with open("../../configs/conv_block.yaml", "r") as f:
    conv_cfg = yaml.safe_load(f)
    basic_conv_blocks = []
    time_spec = None
    freq_spec = None
    stacked = None

    for block in conv_cfg["model"]["blocks"]:
        if block["type"] == "conv_layer_squish_freq":
            freq_spec = block["params"]
        elif block["type"] == "conv_layer_squish_time":
            time_spec = block["params"]
        elif block["type"] == "conv_layer_stacked":
            stacked = block["params"]
        else:
            basic_conv_blocks.append(block["params"])


class ConvSequence(nn.Module):
    def __init__(self, conv_cfg):
        super().__init__()
        basic_conv_blocks = []
        time_spec = None
        freq_spec = None
        stacked = None
        for block in conv_cfg["model"]["blocks"]:
            if block["type"] == "conv_layer_squish_freq":
                freq_spec = block["params"]
            elif block["type"] == "conv_layer_squish_time":
                time_spec = block["params"]
            elif block["type"] == "conv_layer_stacked":
                stacked = block["params"]
            else:
                basic_conv_blocks.append(block["params"])

        self.conv_blocks = nn.ModuleList([
            ConvBlock(block_cfg) for block_cfg in basic_conv_blocks
        ])

        self.freq_conv = ConvBlock(freq_spec) if freq_spec else None
        self.time_conv = ConvBlock(time_spec) if time_spec else None
        self.stacked_conv = ConvBlock(stacked) if stacked else None
        self.linear = nn.Linear(, latent_dim) if latent_dim else None

    def forward(self, x1, x2):
        out1 = x1
        out2 = x2
        for conv in self.conv_blocks:
            out1 = conv(out1)
            out2 = conv(out2)

        out1 = torch.cat([out1, x1], dim=1)  # Residual connection
        out2 = torch.cat([out2, x2], dim=1)  # Residual connection

        freq = self.freq_conv(out1) if self.freq_conv else None
        time = self.time_conv(out2) if self.time_conv else None

        final = torch.cat(
            [freq, time], dim=1) if freq is not None and time is not None else None
        final = self.stacked_conv(
            final) if self.stacked_conv and final is not None else final
        final_fc = torch.flatten(
            final, start_dim=1) if final is not None else None
        return final


class EncoderModel(nn.Module):
    def __init__(self, spec_config, conv_block_configs, inv_spec_config):
        super().__init__()

        self.spec_block = SpectrogramBlock(config["spectrogram"])
        self.conv_blocks = nn.ModuleList([
            ConvBlock(block_cfg) for block_cfg in config["conv_blocks"]
        ])
        self.inv_spec_block = InvSpecBlock(config["inverse_spectrogram"])

    def forward(self, x):
        spec_outputs = self.spec_block(x)
        out = spec_outputs["combined_spec"]
        for conv in self.conv_blocks:
            out = conv(out)
        recon_spec = out
        # Use only 2 channels for inverse
        recon_wave = self.inv_spec_block(recon_spec[:, :2, :, :])
        return {
            "recon_spec": recon_spec,
            "recon_wave": recon_wave,
            **spec_outputs
        }
