"""
Orchestration script to run multiple training configurations.
Usage: python tools/orchestrate_training.py --configs configs/list.txt --epochs 50

This script clones the repo location (assumes running inside container with repo mounted or cloned),
loads config variants from provided file or hard-coded list, launches mlflow server in background
and runs experiments using the existing src/train/train.py entrypoint programmatically.
"""
import torch
from torch.utils.data import DataLoader
from src.models.combined.encoder_decoder import EncoderDecoderModel
from utils.data_loader_gcs import MusicGenreDataset
from utils.load_all_configs import load_all_configs
from src.train.mlflow_runner import run_experiment
import argparse
import os
import subprocess
import time
import sys
from pathlib import Path

# Add project root to PYTHONPATH when running from tools/
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gcs_bucket", required=True,
                   help="GCS bucket name (e.g. gs://my-bucket)")
    p.add_argument("--configs", nargs="*",
                   default=["default"], help="List of config variants to run")
    p.add_argument("--epochs", type=int, default=50,
                   help="Epochs per experiment")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--mlflow_uri", default="http://0.0.0.0:5000",
                   help="MLflow tracking URI")
    p.add_argument("--run_local_mlflow", action="store_true",
                   help="Start a local mlflow server in background")
    return p.parse_args()


def start_mlflow_server(logs_dir: str = "mlruns"):
    # Start mlflow server in background (uses default storage in ./mlruns)
    cmd = [
        "mlflow",
        "server",
        "--host",
        "0.0.0.0",
        "--port",
        "5000",
        "--default-artifact-root",
        os.path.abspath(logs_dir),
    ]
    print("Starting mlflow server:", " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    time.sleep(2)
    return proc


def main():
    args = parse_args()

    if args.run_local_mlflow:
        mlflow_proc = start_mlflow_server()
        print("MLflow server launched (background), give it a few seconds to initialize.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    dataset = MusicGenreDataset(args.gcs_bucket)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    for variant in args.configs:
        print(f"Running variant: {variant}")
        configs = load_all_configs(variant)
        model = EncoderDecoderModel(configs).to(device)

        run_experiment(
            model=model,
            dataloader=dataloader,
            num_epochs=args.epochs,
            lr=1e-4,
            beta_kl=0.001,
            recon_weight=1.0,
            class_weight=0.5,
            device=device,
            experiment_name=f"MusicGenreVAE_{variant}",
            log_dir="logs",
        )

    if args.run_local_mlflow:
        print("Orchestration finished. MLflow server still running in background.")


if __name__ == "__main__":
    main()
