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
        self.fs = gcsfs.GCSFileSystem(
            token=os.environ["GOOGLE_APPLICATION_CREDENTIALS"])
        self.bucket_name = bucket_name
        self.prefix = prefix
        self.sample_rate = sample_rate
        self.clip_length = sample_length  # clip_duration * sample_rate
        self.stride = stride * sample_rate

        # List all mp3 files in GCS
        all_files = self.fs.glob(f"{bucket_name}/{prefix}/**/*.mp3")

        self.audio_files = [f for f in all_files if f.endswith(".mp3")]

        # Extract genre names from path
        self.genres = sorted(
            list({Path(f).parts[-2] for f in self.audio_files}))
        self.genre_to_idx = {genre: idx for idx,
                             genre in enumerate(self.genres)}

        # Only store file paths; clip indices will be computed on demand
        self.clip_map = []
        for audio_path in self.audio_files:
            try:
                with self.fs.open(audio_path, "rb") as f:
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

        # cache for currently loaded file
        self.current_file = None
        self.current_waveform = None
        self.current_sr = None

    def __len__(self):
        return sum(num_clips for _, _, num_clips in self.clip_map)

    def __getitem__(self, idx):
        # Find which file and which clip
        cum = 0
        for audio_path, total_samples, num_clips in self.clip_map:
            if idx < cum + num_clips:
                clip_idx = idx - cum
                start = clip_idx * self.stride
                end = min(start + self.clip_length, total_samples)
                genre = Path(audio_path).parts[-2]
                break
            cum += num_clips
        else:
            raise IndexError(f"Index {idx} out of range")

        # Load audio file if not cached
        if self.current_file != audio_path:
            with self.fs.open(audio_path, "rb") as f:
                audio_bytes = f.read()
            audio_stream = BytesIO(audio_bytes)
            waveform, sr = torchaudio.load(audio_stream)
            self.current_file = audio_path
            self.current_waveform = waveform
            self.current_sr = sr

        waveform = self.current_waveform[:, start:end]

        # FIX 1: Force Mono (Average over channels if stereo)
        # If shape is (2, L), mean(dim=0) -> (L,) -> unsqueeze(0) -> (1, L)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample if needed
        if self.current_sr != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, self.current_sr, self.sample_rate)

        # Pad or trim to exact clip length
        if waveform.size(-1) < self.clip_length:
            pad_amt = self.clip_length - waveform.size(-1)
            waveform = torch.nn.functional.pad(waveform, (0, pad_amt))
        elif waveform.size(-1) > self.clip_length:
            waveform = waveform[:, :self.clip_length]

        # waveform = waveform.unsqueeze(0)
        label = self.genre_to_idx[genre]

        return waveform, label
