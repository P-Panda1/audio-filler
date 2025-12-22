import os
import random
from pathlib import Path
from io import BytesIO

import torch
from torch.utils.data import Dataset
import torchaudio
import gcsfs
from torchcodec.decoders import AudioDecoder


class MusicGenreDataset(Dataset):
    def __init__(
        self,
        bucket_name,
        prefix="music",
        clip_duration=15,     # seconds
        stride=1,             # seconds
        sample_rate=16000,
    ):
        # ===== config =====
        self.bucket_name = bucket_name
        self.prefix = prefix
        self.clip_duration = clip_duration
        self.stride_sec = stride
        self.target_sr = sample_rate
        self.target_len = int(clip_duration * sample_rate)

        # worker-local fs (lazy init)
        self.fs = None

        # ===== scan metadata (main process only) =====
        temp_fs = gcsfs.GCSFileSystem(
            token=os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        )

        print("Scanning GCS for files...")
        all_files = temp_fs.glob(f"{bucket_name}/{prefix}/**/*.mp3")
        self.audio_files = [f for f in all_files if f.endswith(".mp3")]

        self.genres = sorted({Path(f).parts[-2] for f in self.audio_files})
        self.genre_to_idx = {g: i for i, g in enumerate(self.genres)}

        # clip_map entries:
        # (audio_path, total_samples, num_clips, genre_idx)
        self.clip_map = []

        for audio_path in self.audio_files:
            try:
                with temp_fs.open(audio_path, "rb") as f:
                    decoder = AudioDecoder(source=f)
                    sr = decoder.metadata.sample_rate
                    total_samples = int(
                        decoder.metadata.duration_seconds_from_header * sr
                    )
            except Exception:
                continue

            if total_samples < self.target_len:
                continue

            stride_samples = int(self.stride_sec * sr)
            num_clips = max(
                1, (total_samples - self.target_len) // stride_samples + 1
            )

            genre = Path(audio_path).parts[-2]
            genre_idx = self.genre_to_idx[genre]

            self.clip_map.append(
                (audio_path, total_samples, num_clips, genre_idx)
            )

        # prefix sums for O(1) indexing
        self.cum_clips = []
        total = 0
        for _, _, n, _ in self.clip_map:
            total += n
            self.cum_clips.append(total)

        # per-worker cache
        self.current_file = None
        self.current_waveform = None
        self.current_sr = None

        print(f"Dataset ready. {len(self.clip_map)} files, {len(self)} clips.")

    def __len__(self):
        return self.cum_clips[-1] if self.cum_clips else 0

    def _init_fs(self):
        if self.fs is None:
            self.fs = gcsfs.GCSFileSystem(
                token=os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
            )

    def __getitem__(self, idx):
        for _ in range(5):  # bounded retry
            try:
                return self._get_item(idx)
            except Exception:
                idx = random.randint(0, len(self) - 1)

        raise RuntimeError("Too many corrupt samples")

    def _get_item(self, idx):
        self._init_fs()

        # locate file via prefix sums
        file_idx = next(
            i for i, c in enumerate(self.cum_clips) if idx < c
        )
        prev = 0 if file_idx == 0 else self.cum_clips[file_idx - 1]
        clip_idx = idx - prev

        audio_path, _, _, label = self.clip_map[file_idx]

        # load + cache file
        if self.current_file != audio_path:
            with self.fs.open(audio_path, "rb") as f:
                audio_bytes = f.read()
            waveform, sr = torchaudio.load(BytesIO(audio_bytes))

            self.current_file = audio_path
            self.current_waveform = waveform
            self.current_sr = sr

        native_sr = self.current_sr
        stride_samples = int(self.stride_sec * native_sr)

        start = clip_idx * stride_samples
        end = start + int(self.clip_duration * native_sr)

        if start >= self.current_waveform.size(-1):
            raise ValueError("Truncated file")

        waveform = self.current_waveform[:, start:end]

        if waveform.numel() == 0:
            raise ValueError("Empty slice")

        # mono
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # resample
        if native_sr != self.target_sr:
            waveform = torchaudio.functional.resample(
                waveform, native_sr, self.target_sr
            )

        # pad / crop
        if waveform.size(-1) < self.target_len:
            pad = self.target_len - waveform.size(-1)
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        else:
            waveform = waveform[:, :self.target_len]

        return waveform, label
