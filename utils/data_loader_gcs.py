import os
from pathlib import Path
import torch
from torch.utils.data import Dataset
import torchaudio
import gcsfs
from torchcodec.decoders import AudioDecoder
from io import BytesIO


class MusicGenreDataset(Dataset):
    def __init__(self, bucket_name, prefix="music", clip_duration=15, stride=1, sample_rate=16000, sample_length=240000):
        # 1. Setup local FS just for the metadata scan (don't save to self.fs!)
        #    We use this temporarily to map the dataset in the main process.
        temp_fs = gcsfs.GCSFileSystem(
            token=os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))

        self.bucket_name = bucket_name
        self.prefix = prefix
        self.sample_rate = sample_rate
        self.clip_length = sample_length
        self.stride = stride * sample_rate

        # Important: Initialize the worker's FS handle to None
        self.fs = None

        print("Scanning GCS for files (this may take a while)...")
        all_files = temp_fs.glob(f"{bucket_name}/{prefix}/**/*.mp3")
        self.audio_files = [f for f in all_files if f.endswith(".mp3")]

        self.genres = sorted(
            list({Path(f).parts[-2] for f in self.audio_files}))
        self.genre_to_idx = {genre: idx for idx,
                             genre in enumerate(self.genres)}

        self.clip_map = []

        # Use the temporary FS for the initialization scan
        for audio_path in self.audio_files:
            try:
                with temp_fs.open(audio_path, "rb") as f:
                    # Note: We are using torchcodec here to peek at headers quickly
                    decoder = AudioDecoder(source=f)
                    audio_sample_rate = decoder.metadata.sample_rate
                    total_samples = int(
                        decoder.metadata.duration_seconds_from_header * audio_sample_rate)
            except Exception as e:
                print(f"⚠️ Skipping {audio_path}: {e}")
                continue

            if total_samples < self.clip_length:
                continue

            num_clips = max(
                1, (total_samples - self.clip_length) // self.stride + 1)
            self.clip_map.append((audio_path, total_samples, num_clips))

        # Reset cache vars
        self.current_file = None
        self.current_waveform = None
        self.current_sr = None

        print(f"Dataset ready. Found {len(self.clip_map)} valid files.")

    def __len__(self):
        return sum(num_clips for _, _, num_clips in self.clip_map)

    def __getitem__(self, idx):
        # 2. LAZY INITIALIZATION (The Fix)
        # If this is a worker process, self.fs will be None initially.
        # We create a fresh connection here, which is safe inside the worker.
        if self.fs is None:
            self.fs = gcsfs.GCSFileSystem(
                token=os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))

        # Find which file and which clip
        cum = 0
        target_entry = None

        # Optimization: You might want to binary search this if clip_map is huge
        for entry in self.clip_map:
            audio_path, total_samples, num_clips = entry
            if idx < cum + num_clips:
                target_entry = entry
                break
            cum += num_clips
        else:
            raise IndexError(f"Index {idx} out of range")

        audio_path, total_samples, num_clips = target_entry
        clip_idx = idx - cum
        start = clip_idx * self.stride
        end = min(start + self.clip_length, total_samples)
        genre = Path(audio_path).parts[-2]

        # Load audio file if not cached
        # Note: In multi-worker setup, this cache works per-worker.
        if self.current_file != audio_path:
            with self.fs.open(audio_path, "rb") as f:
                audio_bytes = f.read()
            audio_stream = BytesIO(audio_bytes)
            waveform, sr = torchaudio.load(audio_stream)
            self.current_file = audio_path
            self.current_waveform = waveform
            self.current_sr = sr

        # Slicing the cached waveform
        waveform = self.current_waveform[:, start:end]

        # Force Mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample
        if self.current_sr != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, self.current_sr, self.sample_rate)

        # Pad or trim
        if waveform.size(-1) < self.clip_length:
            pad_amt = self.clip_length - waveform.size(-1)
            waveform = torch.nn.functional.pad(waveform, (0, pad_amt))
        elif waveform.size(-1) > self.clip_length:
            waveform = waveform[:, :self.clip_length]

        label = self.genre_to_idx[genre]

        return waveform, label
