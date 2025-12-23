#!/usr/bin/env python3
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from dotenv import load_dotenv
import gcsfs

from utils.data_loader import MusicGenreDataset
from utils.load_all_configs import load_all_configs
from src.blocks.SpectrogramBlock import SpectrogramBlock

# --------------------------------------------------
# Load environment variables
# --------------------------------------------------
load_dotenv()

DATA_BUCKET = os.getenv("DATA_BUCKET")          # gs://audio-filler-1
LOCAL_DATA_DIR = os.getenv("LOCAL_DATA_DIR")    # e.g. /mnt/data
CLIP_DURATION = int(os.getenv("CLIP_DURATION", 15))
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", 16000))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 8))

NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Local + remote paths
LOCAL_OUT_DIR = Path("/tmp/spec_cache")
GCS_OUT_DIR = f"{DATA_BUCKET}/optimised"

LOCAL_OUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------


def main():
    print("📦 Loading dataset...")
    dataset = MusicGenreDataset(
        data_dir="/Users/peeyushpatel/data/project/music",
        clip_duration=CLIP_DURATION,
        stride=1,
        sample_rate=SAMPLE_RATE,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
    )

    print("🎛️ Loading spectrogram block...")
    _, _, spectrogram_config, _ = load_all_configs("large_model")
    spectrogram = SpectrogramBlock(spectrogram_config, DEVICE).to(DEVICE)
    spectrogram.eval()

    fs = gcsfs.GCSFileSystem()
    idx = 0

    print("🚀 Starting preprocessing...")

    with torch.no_grad():
        for waveforms, labels in loader:
            waveforms = waveforms.to(DEVICE)

            spec_dict = spectrogram(waveforms)
            freq = spec_dict["freq_spec"]
            time = spec_dict["time_spec"]

            if freq.dim() == 5:
                freq = freq.squeeze(2)
            if time.dim() == 5:
                time = time.squeeze(2)

            freq_max = freq.abs().amax(dim=[2, 3], keepdim=True)
            time_max = time.abs().amax(dim=[2, 3], keepdim=True)

            freq_norm = freq / (freq_max + 1e-8)
            time_norm = time / (time_max + 1e-8)

            for b in range(freq.shape[0]):
                local_file = f"/{idx:08d}.pt"
                gcs_file = f"{GCS_OUT_DIR}/{idx:08d}.pt"

                if fs.exists(gcs_file):
                    idx += 1
                    continue  # resume-safe

                sample = {
                    "freq_spec": freq_norm[b].cpu(),
                    "time_spec": time_norm[b].cpu(),
                    "freq_max": freq_max[b].cpu(),
                    "time_max": time_max[b].cpu(),
                }

                torch.save(sample, local_file)
                fs.put(str(local_file), gcs_file)
                local_file.unlink()  # free disk

                idx += 1

            if idx % 500 == 0:
                print(f"✅ Saved {idx} samples")

    print(f"🎉 Done. Total samples written: {idx}")


if __name__ == "__main__":
    main()
