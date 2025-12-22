#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import gcsfs
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(
        description="Download music dataset from GCS to local disk (idempotent)"
    )
    p.add_argument(
        "--gcs-bucket",
        required=True,
        help="GCS bucket path (e.g. gs://my-bucket)"
    )
    p.add_argument(
        "--prefix",
        default="music",
        help="Prefix inside bucket (default: music)"
    )
    p.add_argument(
        "--out-dir",
        required=True,
        help="Local directory to store files (e.g. /mnt/data)"
    )
    p.add_argument(
        "--ext",
        default=".mp3",
        help="File extension to download (default: .mp3)"
    )
    return p.parse_args()


def main():
    args = parse_args()

    out_root = Path(args.out_dir).expanduser().resolve()
    music_root = out_root / args.prefix
    music_root.mkdir(parents=True, exist_ok=True)

    fs = gcsfs.GCSFileSystem(
        token=os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    )

    print(f"Scanning {args.gcs_bucket}/{args.prefix} for *{args.ext} files...")
    gcs_files = fs.glob(f"{args.gcs_bucket}/{args.prefix}/**/*{args.ext}")

    if not gcs_files:
        print("No files found. Exiting.")
        return

    print(f"Found {len(gcs_files)} files. Downloading if missing...")

    for gcs_path in tqdm(gcs_files):
        rel_path = gcs_path.replace(f"{args.gcs_bucket}/", "")
        local_path = out_root / rel_path

        # chill out if already exists
        if local_path.exists():
            continue

        local_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with fs.open(gcs_path, "rb") as fsrc, open(local_path, "wb") as fdst:
                fdst.write(fsrc.read())
        except Exception as e:
            print(f"⚠️ Failed to download {gcs_path}: {e}")

    print("Download complete.")


if __name__ == "__main__":
    main()
