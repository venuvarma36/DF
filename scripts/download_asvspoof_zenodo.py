#!/usr/bin/env python3
"""
Automated downloader for ASVspoof 2021 DF/PA parts from Zenodo and keys from asvspoof.org.
Downloads to data/raw/asvspoof2021/, extracts tar.gz archives, and optionally
runs prepare_audio_asvspoof.py to organize into data/audio/{train,val,test}/{real,fake}.

Notes:
- URLs follow the Zenodo pattern for the published records.
- Archives are large (multiple 7–9 GB files). Ensure sufficient disk space and stable network.
- Existing files are skipped to support resume.
"""

import hashlib
import tarfile
import argparse
from pathlib import Path
import requests
from tqdm import tqdm
import subprocess
import sys

# Zenodo record URLs (public)
ZENODO_BASE = "https://zenodo.org/record"

DATASETS = {
    "DF": {
        "record": "4835108",
        "parts": [
            ("ASVspoof2021_DF_eval_part00.tar.gz", "4f2cbae07cf3ede2a1dde3b8d2ee55ea"),
            ("ASVspoof2021_DF_eval_part01.tar.gz", "1578c89ab433c2b60b1dce93bdf8fbec"),
            ("ASVspoof2021_DF_eval_part02.tar.gz", "5497a35f0126e94a1d7a7d26db57b4f7"),
            ("ASVspoof2021_DF_eval_part03.tar.gz", "42b7512ba2943e98a32a53c9608cf03c"),
        ],
        "keys": "https://www.asvspoof.org/asvspoof2021/DF-keys-full.tar.gz",
    },
    "PA": {
        "record": "4834716",
        "parts": [
            ("ASVspoof2021_PA_eval_part00.tar.gz", "78bd4f9178b7508f2b5f78ecc5567335"),
            ("ASVspoof2021_PA_eval_part01.tar.gz", "8b6e8c68cfea31cbb4bc86c943b9ac68"),
            ("ASVspoof2021_PA_eval_part02.tar.gz", "d146e462b932905d13c9b3d5c82d960d"),
            ("ASVspoof2021_PA_eval_part03.tar.gz", "d9602472355b62fb2841a571c789d0e2"),
            ("ASVspoof2021_PA_eval_part04.tar.gz", "194b5a903edca4be1696d476644b4410"),
            ("ASVspoof2021_PA_eval_part05.tar.gz", "1ba1f986db0a33cb9e659c44c41afee1"),
            ("ASVspoof2021_PA_eval_part06.tar.gz", "59c33780525e3b324cd0f84e4ecf08cb"),
        ],
        "keys": "https://www.asvspoof.org/asvspoof2021/PA-keys-full.tar.gz",
    },
    # Add LA here if needed (requires the LA record id and parts).
}


def download(url: str, dest: Path, expected_md5: str | None = None) -> None:
    """Download a file with progress; skip if exists and checksum matches."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and expected_md5:
        if md5(dest) == expected_md5:
            print(f"✅ Skip existing (md5 ok): {dest.name}")
            return
        else:
            print(f"⚠️  Existing file md5 mismatch, re-downloading: {dest.name}")
            dest.unlink()
    elif dest.exists():
        print(f"✅ Skip existing: {dest.name}")
        return

    print(f"📥 Downloading {dest.name} ...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024) as pbar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    if expected_md5:
        digest = md5(dest)
        if digest != expected_md5:
            raise ValueError(f"MD5 mismatch for {dest.name}: {digest} != {expected_md5}")
    print(f"✅ Downloaded: {dest.name}")


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_all(raw_dir: Path) -> None:
    for tar_path in raw_dir.glob("*.tar.gz"):
        print(f"📦 Extracting {tar_path.name} ...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(raw_dir)
    print("✅ Extraction complete")


def run_prepare() -> None:
    print("▶ Running prepare_audio_asvspoof.py ...")
    cmd = [sys.executable, "scripts/prepare_audio_asvspoof.py"]
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Download ASVspoof 2021 DF/PA from Zenodo and prepare audio dataset.")
    parser.add_argument("--subset", choices=list(DATASETS.keys()) + ["all"], default="DF", help="Which subset to download")
    parser.add_argument("--no-prepare", action="store_true", help="Skip organizing into data/audio")
    args = parser.parse_args()

    subsets = DATASETS.keys() if args.subset == "all" else [args.subset]

    for subset in subsets:
        cfg = DATASETS[subset]
        record = cfg["record"]
        raw_dir = Path("data/raw/asvspoof2021")
        raw_dir.mkdir(parents=True, exist_ok=True)

        # Download parts
        for filename, checksum in cfg["parts"]:
            url = f"{ZENODO_BASE}/{record}/files/{filename}?download=1"
            download(url, raw_dir / filename, expected_md5=checksum)

        # Download keys
        keys_url = cfg["keys"]
        keys_dest = raw_dir / keys_url.split("/")[-1]
        download(keys_url, keys_dest)

    # Extract all tar.gz archives
    extract_all(Path("data/raw/asvspoof2021"))

    # Organize into train/val/test unless skipped
    if not args.no_prepare:
        run_prepare()

    print("\n✅ ASVspoof download + organize complete.")
    print("Data ready at data/audio/{train,val,test}/{real,fake}")


if __name__ == "__main__":
    main()
