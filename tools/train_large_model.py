#!/usr/bin/env python3
"""
Train the `large_model` configuration (no MLflow) directly on the pod.

Usage example:
  export GOOGLE_APPLICATION_CREDENTIALS=~/google-account/project-audio-filler-bucket-602aa3031582.json
  python tools/train_large_model.py --gcs-bucket gs://your-bucket --epochs 50 --batch-size 32

This script:
- Sets GOOGLE_APPLICATION_CREDENTIALS (if provided via --service-account)
- Loads the `large_model` configs via utils.load_all_configs
- Builds the dataset from GCS using utils.data_loader_gcs.MusicGenreDataset
- Creates the EncoderDecoderModel and calls train_model

Note: run this on the runpod host directly (no Docker) where Python and dependencies are installed.
"""
from src.train.train_function import train_model
from src.models.combined.encoder_decoder import EncoderDecoderModel
from utils.data_loader_gcs import MusicGenreDataset
# from utils.data_loader import MusicGenreDataset
from utils.load_all_configs import load_all_configs
from torch.utils.data import DataLoader
import torch
import argparse
import os
import sys
from pathlib import Path

# Allow running from repo root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args():
    p = argparse.ArgumentParser(
        description="Train large_model configuration on GCS data (no MLflow)")
    p.add_argument("--data-bucket", required=True,
                   help="GCS bucket path (e.g. gs://my-bucket)")
    p.add_argument("--model-bucket", default=None,
                   help="GCS bucket to upload best model(e.g. gs: // my-bucket)")
    p.add_argument("--service-account", default=None,
                   help="Path to GCP JSON key to export as GOOGLE_APPLICATION_CREDENTIALS")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--beta-kl", type=float, default=0.001)
    p.add_argument("--recon-weight", type=float, default=1.0)
    p.add_argument("--class-weight", type=float, default=0.5)
    p.add_argument("--clip-duration", type=int, default=15,
                   help="Clip duration in seconds")
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--log-dir", default="logs")
    p.add_argument("--val-split", type=float, default=0.1,
                   help="Fraction of dataset to reserve for validation (0.0-1.0)")
    p.add_argument("--create-archive", action="store_true",
                   help="Create a logs tar.gz archive after training")
    p.add_argument("--archive-path", default=None,
                   help="Path to write the logs archive (if not set a timestamped file next to logs/ will be used)")
    p.add_argument("--upload-best-to-gcs", action="store_true",
                   help="Upload the best model to GCS after training (requires google-cloud-storage and a bucket)")
    p.add_argument("--gcs-dest-prefix", default=None,
                   help="Destination prefix inside the GCS bucket where the best model will be uploaded")
    return p.parse_args()


def main():
    args = parse_args()

    if args.service_account:
        sa_path = os.path.expanduser(args.service_account)
        if not os.path.exists(sa_path):
            print(f"Service account file not found: {sa_path}")
            sys.exit(1)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = sa_path
        print(f"Using GOOGLE_APPLICATION_CREDENTIALS={sa_path}")

    device = "cuda" if torch.cuda.is_available() else "mps"
    print(f"Using device: {device}")

    # Build dataset and dataloader
    print("Building dataset from GCS... this may take a moment")
    dataset = MusicGenreDataset(args.data_bucket, prefix="music",
                                clip_duration=args.clip_duration, sample_rate=args.sample_rate)
    # dataset = MusicGenreDataset(
    #     data_dir="/Users/peeyushpatel/data/project/music",
    #     clip_duration=args.clip_duration,  # seconds
    #     stride=1,          # seconds
    #     sample_rate=args.sample_rate
    # )

    # Split into train/val if requested
    val_dataloader = None
    if args.val_split and 0.0 < args.val_split < 1.0:
        total = len(dataset)
        val_size = int(total * args.val_split)
        train_size = total - val_size
        if val_size <= 0 or train_size <= 0:
            print("Dataset too small or val_split too large; not splitting")
        else:
            from torch.utils.data import random_split
            train_ds, val_ds = random_split(dataset, [train_size, val_size])
            dataloader = DataLoader(
                train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=True)
            val_dataloader = DataLoader(
                val_ds, batch_size=args.batch_size, shuffle=False, pin_memory=True)
            print(f"Split dataset: train={train_size}, val={val_size}")
    if val_dataloader is None:
        dataloader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True)

    # Load large_model configs
    configs = load_all_configs("large_model")

    # Instantiate model
    model = EncoderDecoderModel(configs).to(device)

    # Ensure logs directory exists
    os.makedirs(args.log_dir, exist_ok=True)

    # Run training (direct call, no MLflow)
    try:
        train_model(
            model=model,
            dataloader=dataloader,
            num_epochs=args.epochs,
            device=device,
            lr=args.lr,
            beta_kl=args.beta_kl,
            recon_weight=args.recon_weight,
            class_weight=args.class_weight,
            log_interval=10,
            log_dir=args.log_dir,
            upload_best_to_gcs=args.upload_best_to_gcs,
            gcs_bucket=(args.model_bucket.replace('gs://', '')
                        if args.model_bucket else None),
            gcs_dest_prefix=args.gcs_dest_prefix,
            create_archive=args.create_archive,
            archive_path=args.archive_path
        )
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
