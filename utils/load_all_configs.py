import yaml


def load_all_configs():
    configs = []
    with open("configs/encoder_1.yaml", "r") as f:
        encoder_config = yaml.safe_load(f)
        configs.append(encoder_config)
    with open("configs/decoder_1.yaml", "r") as f:
        decoder_config = yaml.safe_load(f)
        configs.append(decoder_config)
    with open("configs/spectrogram_block.yaml", "r") as f:
        spectrogram_config = yaml.safe_load(f)
        configs.append(spectrogram_config)
    with open("configs/inv_spectrogram_block.yaml", "r") as f:
        inv_spectrogram_config = yaml.safe_load(f)
        configs.append(inv_spectrogram_config)
    return configs
