import os
from pathlib import Path
import torch
from torch.utils.data import Dataset
import torchaudio
import gcsfs
from torchcodec.decoders import AudioDecoder
from io import BytesIO


class MusicGenreDataset(Dataset):
    def __init__(self, bucket_name, prefix="music", clip_duration=15, stride=1, sample_rate=16000):
        self.fs = gcsfs.GCSFileSystem(
            token=os.environ["GOOGLE_APPLICATION_CREDENTIALS"])
        self.bucket_name = bucket_name
        self.prefix = prefix
        self.sample_rate = sample_rate
        self.clip_length = clip_duration * sample_rate
        self.stride = stride * sample_rate

        # List all mp3 files in GCS
        all_files = self.fs.ls(f"{bucket_name}/{prefix}")
        self.audio_files = [f for f in all_files if f.endswith(".mp3")]

        # Extract genre names from path
        self.genres = sorted(
            list({Path(f).parts[-2] for f in self.audio_files}))
        self.genre_to_idx = {genre: idx for idx,
                             genre in enumerate(self.genres)}

        # Build clip index
        self.clips = []
        for audio_path in self.audio_files:
            genre = Path(audio_path).parts[-2]
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
                print(
                    f"⚠️ Skipping {audio_path}: too short ({total_samples} samples)")
                continue

            start = 0
            while start < total_samples:
                end = start + self.clip_length
                if end > total_samples:
                    if total_samples - start >= 10 * self.sample_rate:
                        end = total_samples
                        padding = int(self.clip_length - (end - start))
                        self.clips.append(
                            (audio_path, start, end, padding, genre))
                    break
                else:
                    self.clips.append((audio_path, start, end, 0, genre))
                start += self.stride

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        audio_path, start, end, padding, genre = self.clips[idx]

        # Download chunk from GCS
        with self.fs.open(audio_path, "rb") as f:
            audio_bytes = f.read()
        audio_stream = BytesIO(audio_bytes)
        waveform, sr = torchaudio.load(
            audio_stream, frame_offset=int(start), num_frames=int(end - start))

        if waveform.numel() == 0:
            raise IndexError(
                f"Empty waveform for {audio_path} at {start}:{end}")

        waveform = torchaudio.functional.resample(
            waveform, sr, self.sample_rate)[0]

        target_len = self.clip_length
        if waveform.size(-1) < target_len:
            pad_amt = target_len - waveform.size(-1)
            waveform = torch.nn.functional.pad(waveform, (0, pad_amt))
        elif waveform.size(-1) > target_len:
            waveform = waveform[:, :target_len]

        waveform = waveform.unsqueeze(0)
        label = self.genre_to_idx[genre]

        return waveform, label
