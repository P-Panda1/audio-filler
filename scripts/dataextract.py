import os
import re
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm.auto import tqdm
import browsercookie


# map genres -> pages (same as yours)
genre_urls = {
    "blues": "https://freemusicarchive.org/genre/Blues?pageSize=50&page=1&search-genre=Blues&sort=_score&d=0",
    "jazz": "https://freemusicarchive.org/genre/Jazz?pageSize=50&page=1&search-genre=Jazz&sort=_score&d=0",
    "country": "https://freemusicarchive.org/genre/Country?pageSize=50&page=1&search-genre=Country&sort=_score&d=0",
    "pop": "https://freemusicarchive.org/genre/Pop?pageSize=50&page=1&search-genre=Pop&sort=_score&d=0",
    "lofi": "https://freemusicarchive.org/genre/Lo-fi-Instrumental?pageSize=20&page=1&search-genre=Lo-fi%20Instrumental&sort=_score&d=0",
    "rock-garage": "https://freemusicarchive.org/genre/Garage?pageSize=50&page=1&search-genre=Garage&sort=_score&d=0",
    "rock-goth": "https://freemusicarchive.org/genre/Goth?pageSize=50&page=1&search-genre=Goth&sort=_score&d=0",
    "rock-industrial": "https://freemusicarchive.org/genre/Industrial?pageSize=50&page=1&search-genre=Industrial&sort=_score&d=0",
    "rock-krautrock": "https://freemusicarchive.org/genre/Krautrock?pageSize=50&page=1&search-genre=Krautrock&sort=_score&d=0",
    "rock-punk": "https://freemusicarchive.org/genre/Punk?pageSize=50&page=1&search-genre=Punk&sort=_score&d=0",
    "metal": "https://freemusicarchive.org/genre/Metal?pageSize=50&page=1&search-genre=Metal&sort=_score&d=0",
    "rnb": "https://freemusicarchive.org/genre/Soul-RB?pageSize=50&page=1&search-genre=Soul-RnB&sort=_score&d=0",
    "folk": "https://freemusicarchive.org/genre/Folk?pageSize=50&page=1&search-genre=Folk&sort=_score&d=0",
    "classical": "https://freemusicarchive.org/genre/Classical?pageSize=50&page=1&search-genre=Classical&sort=_score&d=0",
    "hiphop": "https://freemusicarchive.org/genre/Hip-Hop?pageSize=50&page=1&search-genre=Hip-Hop&sort=_score&d=0",
}

base_dir = os.path.expanduser("~/data/project/music")
os.makedirs(base_dir, exist_ok=True)

session = requests.Session()
session.cookies = browsercookie.chrome()
session.headers.update({"User-Agent": "Mozilla/5.0", "Accept": "*/*"})

metadata = []

for genre, page_url in genre_urls.items():
    print(f"\nProcessing genre: {genre}")
    genre_dir = os.path.join(base_dir, genre)
    os.makedirs(genre_dir, exist_ok=True)

    # remove old mp3s if you want
    for f in os.listdir(genre_dir):
        if f.endswith(".mp3"):
            os.remove(os.path.join(genre_dir, f))

    # fetch genre page
    r = session.get(page_url, timeout=20)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    # find anchors whose href looks like /track/<slug>/download
    links = soup.find_all("a", {"title": "Download"})

    for idx, a in enumerate(tqdm(links, desc=genre)):
        download_url = a.get("data-url").replace("downloadOverlay", "download")

        # extract the track slug between /track/ and /download
        try:
            track_slug = download_url.split("/track/")[1].split("/download")[0]
        except Exception:
            track_slug = f"track-{genre}-{idx+1}"

        # proposed output filename (use slug so it's readable)
        filename = f"track-{genre}-{idx+1}.mp3"
        filepath = os.path.join(genre_dir, filename)

        try:
            # set Referer to the genre page (sometimes required)
            with session.get(download_url, stream=True, timeout=30,
                             headers={"Referer": page_url}, allow_redirects=True) as dl:
                dl.raise_for_status()

                ctype = dl.headers.get("Content-Type", "").lower()
                # If server returned HTML, likely blocked or needs JS — skip and warn
                if "text/html" in ctype:
                    print(
                        f"Skipping {download_url} — server returned HTML (possible block).")
                    continue

                # try to get filename from Content-Disposition if present
                # disp = dl.headers.get("content-disposition", "")
                # m = re.search(r'filename="?([^";]+)"?', disp)
                # if m:
                #     filename = m.group(1)
                #     filepath = os.path.join(genre_dir, filename)

                total = int(dl.headers.get("Content-Length", 0))
                with open(filepath, "wb") as out_f:
                    with tqdm(total=total, unit="B", unit_scale=True, desc=filename, leave=False) as pbar:
                        for chunk in dl.iter_content(chunk_size=8192):
                            if not chunk:
                                continue
                            out_f.write(chunk)
                            pbar.update(len(chunk))

            metadata.append({
                "file_name": filename,
                "track_slug": track_slug,
                "download_url": download_url,
            })

        except Exception as e:
            print(f"Failed to download {download_url}: {e}")

# save metadata
csv_path = os.path.join(base_dir, "metadata.csv")
pd.DataFrame(metadata).to_csv(csv_path, index=False)
print(f"\nMetadata saved at {csv_path}")
