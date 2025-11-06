import os
import yaml


def load_all_configs():
    # Get absolute path to the project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_dir = os.path.join(project_root, "configs")

    def load_yaml(filename):
        with open(os.path.join(config_dir, filename), "r") as f:
            return yaml.safe_load(f)

    configs = [
        load_yaml("encoder_1.yaml"),
        load_yaml("decoder_1.yaml"),
        load_yaml("spectrogram.yaml"),
        load_yaml("invspec.yaml"),
    ]
    return configs
