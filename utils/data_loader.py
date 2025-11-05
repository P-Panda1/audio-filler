import os
from pathlib import Path
import torch
from torch.utils.data import Dataset
import torchaudio
from torchcodec.decoders import AudioDecoder


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
            # info = torchaudio.info(audio_file)
            # audio_sample_rate = info.sample_rate
            # total_samples = info.num_frames
            try:
                decoder = AudioDecoder(source=audio_file)
                audio_sample_rate = decoder.metadata.sample_rate
                total_samples = decoder.metadata.duration_seconds_from_header * audio_sample_rate

            except Exception as e:
                print(f"⚠️ Skipping {audio_file}: {e}")
                continue
            start = 0
            while start < total_samples:
                # clip_duration in seconds
                end = start + (audio_sample_rate * clip_duration)
                if end > total_samples:
                    if total_samples - start >= 10 * sample_rate:  # Check if at least 10s
                        end = total_samples
                        padding = self.clip_length - (end - start)
                        self.clips.append(
                            (audio_file, start, end, padding, genre))
                    break
                else:
                    self.clips.append((audio_file, start, end, 0, genre))
                start += self.stride

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        audio_file, start, end, padding, genre = self.clips[idx]
        waveform, sr = torchaudio.load(
            audio_file, frame_offset=start, num_frames=end-start)
        waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)[
            0]  # Convert to mono

        if padding > 0:
            waveform = torch.nn.functional.pad(waveform, (0, padding))

        # Add channel dimension
        waveform = waveform.unsqueeze(0)
        label = self.genre_to_idx[genre]

        return waveform, label
