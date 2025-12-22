import os
from pathlib import Path
import torch
from torch.utils.data import Dataset
import torchaudio
# from torchcodec.decoders import AudioDecoder


class MusicGenreDataset(Dataset):
    def __init__(self, data_dir, clip_duration=15, stride=1, sample_rate=16000):
        self.data_dir = Path(os.path.expanduser(str(data_dir)))
        self.clip_length = clip_duration * sample_rate
        self.stride = stride * sample_rate
        self.sample_rate = sample_rate

        # Get genre labels
        self.genres = sorted(
            [d.name for d in self.data_dir.iterdir() if d.is_dir()])
        self.genre_to_idx = {genre: idx for idx,
                             genre in enumerate(self.genres)}

        # Collect audio files and their genres
        self.audio_files = []
        for genre in self.genres:
            genre_dir = self.data_dir / genre
            for audio_file in genre_dir.glob("*.mp3"):
                self.audio_files.append((audio_file, genre))

        # Precompute clip segments
        self.clips = []
        for audio_file, genre in self.audio_files:
            # try:
            #     # info = torchaudio.backend.sox_io_backend.info(audio_file)
            #     # audio_sample_rate = info.sample_rate
            #     # total_samples = info.num_frames

            #     # decoder = AudioDecoder(source=audio_file)
            #     # audio_sample_rate = decoder.metadata.sample_rate
            #     # total_samples = int(
            #     #     decoder.metadata.duration_seconds_from_header * audio_sample_rate
            #     # )
            # except Exception as e:
            #     print(f"⚠️ Skipping {audio_file}: {e}")
            #     continue

            try:
                waveform, audio_sample_rate = torchaudio.load(audio_file)
                total_samples = waveform.shape[-1]
            except Exception as e:
                print(f"⚠️ Skipping {audio_file}: {e}")
                continue

            if total_samples < self.clip_length:
                print(
                    f"⚠️ Skipping {audio_file}: too short ({total_samples} samples)")
                continue

            start = 0
            while start + self.clip_length <= total_samples:
                end = start + self.clip_length
                self.clips.append((audio_file, start, end, genre))
                start += self.stride

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        audio_file, start, end, genre = self.clips[idx]
        waveform, sr = torchaudio.load(
            audio_file,
            frame_offset=int(start),
            num_frames=int(end - start),
        )

        if waveform.numel() == 0:
            waveform = torch.zeros((1, self.clip_length))

        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sr, self.sample_rate
            )

        # Convert to mono
        waveform = waveform.mean(dim=0, keepdim=True)

        # Pad / trim
        if waveform.shape[-1] < self.clip_length:
            waveform = torch.nn.functional.pad(
                waveform, (0, self.clip_length - waveform.shape[-1])
            )
        else:
            waveform = waveform[..., : self.clip_length]

        # Add channel dimension
        waveform = waveform.unsqueeze(0)
        label = self.genre_to_idx[genre]

        return waveform, label
