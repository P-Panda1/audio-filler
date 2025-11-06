from pathlib import Path
from pydub import AudioSegment

data_dir = Path("~/data/project/music").expanduser()

for genre_dir in data_dir.iterdir():
    if not genre_dir.is_dir():
        continue
    for mp3_file in genre_dir.glob("*.mp3"):
        wav_path = mp3_file.with_suffix(".wav")
        if not wav_path.exists():
            try:
                audio = AudioSegment.from_mp3(mp3_file)
                audio.export(wav_path, format="wav")
                print(f"Converted {mp3_file.name} → {wav_path.name}")
            except Exception as e:
                print(f"⚠️ Skipping {mp3_file}: {e}")
