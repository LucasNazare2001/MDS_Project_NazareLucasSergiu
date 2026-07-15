#!/usr/bin/env python
"""
Download a balanced subset of the SynthWildX dataset (GRIP-UniNA) into real/ and
fake/ folders, ready for evaluate.py.

SynthWildX contains images collected from X/Twitter:
  - typ == 'real'                           -> real images
  - typ in {dalle3, midjourney_v5, firefly} -> AI-generated images (fake)

Images are downloaded from pbs.twimg.com, so run this on YOUR machine with open
internet access.

Usage:
    python fetch_synthwildx_subset.py --n 50 --out dataset

Result:
    dataset/
    |-- real/   (50 real images)
    |-- fake/   (50 AI images, split across the 3 generators)

The script is resumable: re-run it to skip already-downloaded files.
"""
import os
import io
import csv
import time
import argparse
import urllib.request
from urllib.error import HTTPError, URLError

RAW_LIST = ("https://raw.githubusercontent.com/grip-unina/"
            "ClipBased-SyntheticImageDetection/main/data/synthwildx/list.csv")
LOCAL_LIST = os.path.join(os.path.dirname(__file__), "synthwildx_list.csv")

HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) "
                         "AppleWebKit/537.36 (KHTML, like Gecko) "
                         "Chrome/120.0 Safari/537.36"}
FAKE_TYPES = ("dalle3", "midjourney_v5", "firefly")


def load_list():
    # read list.csv from the local copy, or fetch it from GitHub raw
    if os.path.isfile(LOCAL_LIST):
        with open(LOCAL_LIST, newline="") as fh:
            return list(csv.DictReader(fh))
    print("local list.csv not found, downloading from GitHub...")
    req = urllib.request.Request(RAW_LIST, headers=HEADERS)
    with urllib.request.urlopen(req) as resp:
        text = resp.read().decode("utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def pick(rows, n_per_class, seed):
    # pick n real and n fake rows (fake split evenly across the 3 generators)
    import random
    rnd = random.Random(seed)

    real = [r for r in rows if r["typ"] == "real"]
    rnd.shuffle(real)
    real = real[:n_per_class]

    per_gen = n_per_class // len(FAKE_TYPES)
    extra = n_per_class - per_gen * len(FAKE_TYPES)
    fake = []
    for i, g in enumerate(FAKE_TYPES):
        pool = [r for r in rows if r["typ"] == g]
        rnd.shuffle(pool)
        take = per_gen + (1 if i < extra else 0)
        fake += pool[:take]
    return real, fake


def download(url, dest):
    if os.path.isfile(dest):
        return "skip"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
            with open(dest, "wb") as fh:
                fh.write(data)
            return "ok"
        except HTTPError as e:
            return f"http{e.code}"           # image removed from X -> give up
        except (URLError, TimeoutError):
            time.sleep(1.5)                   # transient -> retry
    return "fail"


def grab(rows, folder, out_root):
    ok = 0
    for i, r in enumerate(rows, 1):
        ext = os.path.splitext(r["filename"])[1] or ".jpg"
        dest = os.path.join(out_root, folder, f"{folder}_{i:03d}{ext}")
        status = download(r["url"], dest)
        if status in ("ok", "skip"):
            ok += 1
        print(f"  [{folder}] {i:3d}/{len(rows)}  {status}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="images per class (default 50)")
    ap.add_argument("--out", default="dataset", help="output folder")
    ap.add_argument("--seed", type=int, default=0, help="random seed (reproducible)")
    args = ap.parse_args()

    rows = load_list()
    real, fake = pick(rows, args.n, args.seed)
    print(f"Selected {len(real)} real and {len(fake)} fake. "
          f"Downloading into '{args.out}/'...\n")

    n_real = grab(real, "real", args.out)
    n_fake = grab(fake, "fake", args.out)

    print(f"\nDone. Downloaded {n_real} real and {n_fake} fake into '{args.out}/'.")
    if n_real < args.n or n_fake < args.n:
        print("Note: some images are no longer available on X. "
              "Re-run the script (raise --n or change --seed) to complete them.")
    print(f"\nNow run:  python evaluate.py {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
