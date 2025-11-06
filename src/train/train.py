from src.models.combined.encoder_decoder import EncoderDecoderModel
from utils.data_loader import MusicGenreDataset
from torch.utils.data import DataLoader
from src.train.mlflow_runner import run_experiment
import torch

# Load configs
from utils.load_all_configs import load_all_configs
configs = load_all_configs()

# Setup dataset
dataset = MusicGenreDataset("~/data/project/music")
dataloader = DataLoader(dataset, batch_size=500, shuffle=True)

# Model
model = EncoderDecoderModel(configs)

device = "cuda" if torch.cuda.is_available() else "mps"

# Run experiment
run_experiment(
    model=model,
    dataloader=dataloader,
    num_epochs=1,
    lr=1e-4,
    beta_kl=0.001,
    recon_weight=1.0,
    class_weight=0.5,
    device=device,
    experiment_name="MusicGenreVAE",
)
