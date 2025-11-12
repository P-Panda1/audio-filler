import torch
import torch.nn as nn
import torch.nn.functional as F
from src.blocks.ConvBlock import ConvBlock
from functools import reduce
import operator


def parse_encoder_cfg(conv_cfg):
    basic_conv_blocks = []
    secondary_conv_blocks = []
    time_spec = None
    freq_spec = None
    merge_node = None
    for block in conv_cfg["model"]["blocks"]:
        if block["type"] == "squish_freq":
            freq_spec = block["params"]
        elif block["type"] == "squish_time":
            time_spec = block["params"]
        elif block["type"] == "merge_node":
            merge_node = block["params"]
        elif block["type"] == "layer1":
            basic_conv_blocks.append(block["params"])
        elif block["type"] == "secondarylayer":
            secondary_conv_blocks.append(block["params"])

    return basic_conv_blocks, secondary_conv_blocks, freq_spec, time_spec, merge_node


class ConvFirstBranch(nn.Module):
    def __init__(self, basic_conv_blocks, freq_spec, time_spec):
        super().__init__()

        self.conv_blocks = nn.ModuleList([
            ConvBlock(block_cfg) for block_cfg in basic_conv_blocks
        ])

        self.residual_stride = reduce(operator.mul,
                                      [block_cfg.get("stride", 1)
                                       for block_cfg in basic_conv_blocks],
                                      1)

        self.freq_conv = ConvBlock(freq_spec) if freq_spec else None
        self.time_conv = ConvBlock(time_spec) if time_spec else None

        self.residual_conv = nn.Conv2d(3, 8, kernel_size=1, stride=1)

    def forward(self, x1, x2):
        out1 = x1
        out2 = x2
        for conv in self.conv_blocks:
            out1 = conv(out1)
            out2 = conv(out2)

        residual1 = nn.AvgPool2d(
            kernel_size=self.residual_stride,
            stride=self.residual_stride
        )(x1)

        residual2 = nn.AvgPool2d(
            kernel_size=self.residual_stride,
            stride=self.residual_stride
        )(x2)
        # Match channels dynamically
        if residual1.size(1) != out1.size(1):
            residual1 = self.residual_conv(residual1)

        if residual2.size(1) != out2.size(1):
            residual2 = self.residual_conv(residual2)

        out1 = out1 + residual1  # Residual connection
        out2 = out2 + residual2  # Residual connection

        freq = self.freq_conv(out1) if self.freq_conv else None
        time = self.time_conv(out2) if self.time_conv else None

        final = torch.cat(
            [freq, time], dim=1) if freq is not None and time is not None else None

        return final


class ConvFinalBranch(nn.Module):
    def __init__(self, secondary_conv_blocks):
        super().__init__()
        self.conv_blocks = nn.ModuleList([
            ConvBlock(block_cfg) for block_cfg in secondary_conv_blocks
        ])
        self.residual_stride = reduce(operator.mul,
                                      [block_cfg.get("stride", 1)
                                       for block_cfg in secondary_conv_blocks],
                                      1)
        self.residual_pool = nn.AvgPool2d(
            kernel_size=self.residual_stride,
            stride=self.residual_stride
        )
        self.residual_conv = nn.Conv2d(
            in_channels=8,
            out_channels=2,
            kernel_size=1,
            stride=1
        )
        self.final_fc_layer = nn.Linear(2000, 250)

    def forward(self, x):
        final = x
        for conv in self.conv_blocks:
            final = conv(final)
        # Add Avg Pooling Residual with stride 10 to account for conv downsampling
        residual = self.residual_pool(x)

        residual = self.residual_conv(residual)

        final = final + residual
        # Flatten all dimensions except batch
        final = final.view(final.size(0), -1)
        final = self.final_fc_layer(final)
        return final


class EncoderModel1(nn.Module):
    def __init__(self, conv_block_configs, latent_dim=500):
        super().__init__()

        basic_conv_blocks, \
            secondary_conv_blocks, \
            freq_spec, time_spec, \
            merge_node = parse_encoder_cfg(
                conv_block_configs)
        self.branch_1 = nn.ModuleList([
            ConvFirstBranch(basic_conv_blocks, freq_spec, time_spec) for _ in range(8)
        ])
        self.branch_2 = nn.ModuleList([
            ConvFinalBranch(secondary_conv_blocks) for _ in range(4)
        ])

        self.merge_node = ConvBlock(merge_node)

        self.latent_dim = latent_dim
        self.linear1 = nn.Linear(4 * 250, latent_dim * 2)
        self.relu = nn.ReLU()
        self.fc_mu = nn.Linear(latent_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim * 2, latent_dim)

    def forward(self, freq, time):
        branch_1_outputs = []
        for branch in self.branch_1:
            branch_1_outputs.append(branch(freq, time))

        merged = torch.cat(branch_1_outputs, dim=1)
        merged = self.merge_node(merged)
        chunks = torch.chunk(merged, 4, dim=1)
        branch_2_outputs = []
        for i, branch in enumerate(self.branch_2):
            branch_2_outputs.append(branch(chunks[i]))
        branch_2 = torch.stack(branch_2_outputs, dim=1)
        combined = branch_2.view(branch_2.size(0), -1)
        combined = self.linear1(combined)
        combined = self.relu(combined)

        mu = self.fc_mu(combined)
        logvar = self.fc_logvar(combined)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
