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
        # 1. Lazy Init for Worker Processes
        if self.fs is None:
            self.fs = gcsfs.GCSFileSystem(
                token=os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))

        # Find which file and clip corresponds to this index
        cum = 0
        target_entry = None
        for entry in self.clip_map:
            audio_path, duration_sec, num_clips = entry
            if idx < cum + num_clips:
                target_entry = entry
                break
            cum += num_clips

        # Safety fallback if index logic fails
        if target_entry is None:
            return self._get_random_valid_sample()

        audio_path, _, _ = target_entry
        clip_idx = idx - cum
        genre = Path(audio_path).parts[-2]

        try:
            # 2. Load Audio (Cached)
            if self.current_file != audio_path:
                with self.fs.open(audio_path, "rb") as f:
                    audio_bytes = f.read()
                audio_stream = BytesIO(audio_bytes)
                waveform, sr = torchaudio.load(audio_stream)

                self.current_file = audio_path
                self.current_waveform = waveform
                self.current_sr = sr

            # 3. Calculate Indices (Seconds -> Native Samples)
            native_sr = self.current_sr
            start_sec = clip_idx * self.stride_sec
            start_native = int(start_sec * native_sr)
            end_native = int((start_sec + self.clip_duration) * native_sr)

            # 4. CRITICAL CHECK: Is the file actually shorter than the metadata claimed?
            # If the slice is out of bounds, the file is truncated/corrupt.
            if start_native >= self.current_waveform.size(-1):
                raise ValueError(f"File truncated: {audio_path}")

            # Slice raw audio
            waveform = self.current_waveform[:, start_native:end_native]

            # 5. Check for Empty/Zero-length Tensor BEFORE Resampling
            if waveform.numel() == 0 or waveform.size(-1) == 0:
                raise ValueError(f"Empty slice extracted from {audio_path}")

            # 6. Process Audio
            # Force Mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Resample
            if native_sr != self.target_sr:
                waveform = torchaudio.functional.resample(
                    waveform, native_sr, self.target_sr)

            # Pad if slightly short after resampling
            if waveform.size(-1) < self.target_len:
                pad_amt = self.target_len - waveform.size(-1)
                waveform = torch.nn.functional.pad(waveform, (0, pad_amt))
            elif waveform.size(-1) > self.target_len:
                waveform = waveform[:, :self.target_len]

            label = self.genre_to_idx[genre]
            return waveform, label

        except Exception as e:
            # === THE SKIP LOGIC ===
            # Instead of returning zeros, we pick a random new index and try again.
            print(
                f"⚠️ Bad clip at index {idx} ({audio_path}): {e}. Retrying with new sample...")
            return self._get_random_valid_sample()

    def _get_random_valid_sample(self):
        import random
        # Pick a random index and recursively call __getitem__
        # We limit recursion depth implicitly by hoping the dataset isn't 100% corrupt.
        new_idx = random.randint(0, len(self) - 1)
        return self.__getitem__(new_idx)
