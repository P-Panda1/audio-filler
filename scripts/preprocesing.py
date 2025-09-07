import torch
import numpy as np
from torch.utils.data import Dataset
import os
import pandas as pd
from pydub import AudioSegment


def make_csv_from_mp3(input_dir, output_csv, stride=5, clip_len=10):
    rows = []

    for fname in os.listdir(input_dir):
        if not fname.endswith(".mp3"):
            continue

        file_path = os.path.join(input_dir, fname)
        audio = AudioSegment.from_mp3(file_path)
        duration = len(audio) / 1000.0  # convert ms → seconds

        start = 0
        while start + clip_len <= duration:
            label_start = start
            label_end = start + clip_len

            before_start = max(0, label_start - clip_len)
            before_end = label_start

            after_start = label_end
            after_end = label_end + clip_len

            # Only include if both before & after windows are valid 10s
            if before_end - before_start == clip_len and after_end <= duration:
                rows.append({
                    "file": fname,
                    "label_start": label_start,
                    "label_end": label_end,
                    "before_start": before_start,
                    "before_end": before_end,
                    "after_start": after_start,
                    "after_end": after_end
                })

            start += stride

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"Saved metadata CSV with {len(df)} rows to {output_csv}")

# Example usage
# make_csv_from_mp3("data/mp3s", "clips_metadata.csv")


class AudioWindowDataset(Dataset):
    def __init__(self, csv_file, audio_dir, target_sr=16000):
        import pandas as pd
        self.df = pd.read_csv(csv_file)
        self.audio_dir = audio_dir
        self.target_sr = target_sr

    def __len__(self):
        return len(self.df)

    def _load_clip(self, fname, start, end):
        path = os.path.join(self.audio_dir, fname)
        audio = AudioSegment.from_mp3(path)
        clip = audio[start * 1000:end * 1000]  # slice in ms
        clip = clip.set_frame_rate(self.target_sr).set_channels(1)
        samples = np.array(clip.get_array_of_samples()).astype(np.float32)
        samples /= np.iinfo(clip.array_type).max  # normalize to [-1,1]
        return torch.tensor(samples)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file = row["file"]

        label = self._load_clip(file, row["label_start"], row["label_end"])
        before = self._load_clip(file, row["before_start"], row["before_end"])
        after = self._load_clip(file, row["after_start"], row["after_end"])

        return {
            "before": before,
            "label": label,
            "after": after
        }


# Example usage
dataset = AudioWindowDataset("clips_metadata.csv", "data/mp3s")
loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

for batch in loader:
    print(batch["label"].shape)  # [batch, samples]
    break
