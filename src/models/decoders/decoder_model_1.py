import torch
import torch.nn as nn
from src.blocks.ConvBlock import ConvBlock
from src.blocks.ConvTransposeBlock import ConvTransposeBlock


def parse_decoder_cfg(trans):
    blocks = trans['model']['blocks']
    length = len(blocks)
    pair_blocks = []
    for i in range(0, length - 2, 2):
        pair_blocks.append((blocks[i]['params'], blocks[i + 1]['params']))
    final_block = blocks[-1]['params']
    return pair_blocks, final_block


class TransposeLayer(nn.Module):
    def __init__(self, conv_layer, trans_layer):
        super().__init__()

        self.mf = trans_layer['in_channels'] / conv_layer['out_channels']

        self.conv_blocks = nn.ModuleList([
            ConvBlock(conv_layer) for _ in range(int(self.mf))
        ])
        self.conv_transpose_block = ConvTransposeBlock(trans_layer)

    def forward(self, x):
        conv_outs = []
        for conv in self.conv_blocks:
            conv_outs.append(conv(x))
        concat_out = torch.cat(conv_outs, dim=1)
        final = self.conv_transpose_block(concat_out)
        return final


class DecoderModel1(nn.Module):
    def __init__(self, trans_config, latent_dim=512, class_size=15):
        super().__init__()

        self.Linear1 = nn.Linear(latent_dim, 4000)
        pair_blocks, final_block = parse_decoder_cfg(trans_config)
        self.transpose_layers = nn.Sequential(
            *[TransposeLayer(conv_layer, trans_layer)
              for conv_layer, trans_layer in pair_blocks])
        self.final_layer = ConvBlock(final_block)

        self.activation = nn.ReLU()
        self.classifier = nn.Linear(latent_dim, class_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.Linear1(x)
        recon = out.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, 4000)
        for layer in self.transpose_layers:
            recon = layer(recon)
        recon = self.final_layer(recon)

        class_out = self.classifier(x)
        class_out = self.softmax(class_out)
        return recon, class_out
