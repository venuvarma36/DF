#!/usr/bin/env python3
"""
Prepare ASVspoof 2021 audio dataset from available EVAL parts using keys.
- Reads labels from keys-full trial_metadata.txt (LA/PA/DF)
- Finds matching audio files under data/raw/asvspoof2021
- Splits into train/val/test (70/15/15) and copies to data/audio

This is a pragmatic fallback when only EVAL audio is present (no train/dev archives).
"""
from __future__ import annotations
import os
from pathlib import Path
import random
import shutil
from typing import Dict, List, Tuple

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("prep_asvspoof_keys")

RAW_BASE = Path("data/raw")
ASVSPOOF_BASE = RAW_BASE / "asvspoof2021"
KEYS_ROOTS = {
    "LA": RAW_BASE / "LA-keys-full" / "keys" / "LA" / "CM" / "trial_metadata.txt",
    "PA": RAW_BASE / "PA-keys-full" / "keys" / "PA" / "CM" / "trial_metadata.txt",
    "DF": RAW_BASE / "DF-keys-full" / "keys" / "DF" / "CM" / "trial_metadata.txt",
}

OUTPUT = Path("data/audio")
SPLIT_RATIOS = (0.70, 0.15, 0.15)
RANDOM_SEED = 42


def parse_trial_metadata(file_path: Path) -> Dict[str, str]:
    """Parse trial_metadata.txt -> mapping from utterance ID to label ('real'/'fake').
    The standard format has the utterance ID in the 2nd token and label near the end.
    We'll robustly look for 'bonafide' or 'spoof' tokens.
    """
    mapping: Dict[str, str] = {}
    if not file_path.exists():
        logger.warning(f"Keys file not found: {file_path}")
        return mapping

    with file_path.open('r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split()
            # Find label token
            label = None
            for tok in reversed(parts):
                if tok.lower() in ("bonafide", "spoof"):
                    label = "real" if tok.lower() == "bonafide" else "fake"
                    break
            # Utterance id heuristics: prefer 2nd token; fallback to token containing "_E_"
            utt = None
            if len(parts) >= 2:
                utt = parts[1]
            if not utt:
                for tok in parts:
                    if "_E_" in tok:
                        utt = tok
                        break
            if utt and label:
                mapping[utt] = label
    logger.info(f"Parsed {len(mapping):,} keys from {file_path}")
    return mapping


def collect_audio_files(subset: str) -> Dict[str, Path]:
    """Collect all audio files for a subset (LA/PA/DF) under asvspoof2021/<subset>/**.
    Returns map {stem: path}.
    """
    subset_dir = ASVSPOOF_BASE / subset
    if not subset_dir.exists():
        logger.warning(f"Subset dir missing: {subset_dir}")
        return {}
    audio_map: Dict[str, Path] = {}
    for ext in (".flac", ".wav"):
        for p in subset_dir.rglob(f"*{ext}"):
            audio_map[p.stem] = p
    logger.info(f"Found {len(audio_map):,} audio files for {subset}")
    return audio_map


def build_dataset() -> Tuple[List[Tuple[Path, str]], List[Tuple[Path, str]], List[Tuple[Path, str]]]:
    """Create (train, val, test) lists of (path, label)."""
    paired: List[Tuple[Path, str]] = []
    for subset in ("LA", "PA", "DF"):
        keys_file = KEYS_ROOTS.get(subset)
        keys = parse_trial_metadata(keys_file) if keys_file else {}
        if not keys:
            continue
        audio_map = collect_audio_files(subset)
        # Pair available ones
        matched = 0
        for utt, label in keys.items():
            if utt in audio_map:
                paired.append((audio_map[utt], label))
                matched += 1
        logger.info(f"{subset}: matched {matched:,} labeled files")

    # Split
    random.seed(RANDOM_SEED)
    random.shuffle(paired)
    n = len(paired)
    if n == 0:
        raise RuntimeError("No labeled audio files matched. Ensure eval parts and keys are extracted.")
    n_train = int(n * SPLIT_RATIOS[0])
    n_val = int(n * SPLIT_RATIOS[1])
    train = paired[:n_train]
    val = paired[n_train:n_train + n_val]
    test = paired[n_train + n_val:]
    logger.info(f"Split sizes -> train: {len(train):,}, val: {len(val):,}, test: {len(test):,}")
    return train, val, test


def move_split(split: List[Tuple[Path, str]], split_name: str):
    """Move (not copy) audio files to destination split folder.
    Moving saves storage by avoiding duplication.
    """
    for path, label in split:
        out_dir = OUTPUT / split_name / label
        out_dir.mkdir(parents=True, exist_ok=True)
        dst = out_dir / path.name
        if not dst.exists():
            shutil.move(str(path), str(dst))
        else:
            logger.info(f"Skipping (already exists): {dst}")
            # Remove source if destination already exists
            if path.exists():
                path.unlink()


def cleanup_output():
    """Remove existing output folders to avoid conflicts."""
    if OUTPUT.exists():
        logger.info(f"Cleaning up existing output folder: {OUTPUT}")
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True, exist_ok=True)


def main():
    logger.info("Preparing ASVspoof dataset from keys + eval audio (MOVING files to save storage)...")
    cleanup_output()
    train, val, test = build_dataset()
    logger.info("Moving files to data/audio (this saves storage vs copying)...")
    move_split(train, "train")
    move_split(val, "val")
    move_split(test, "test")
    logger.info(f"✅ Done. Audio dataset in {OUTPUT}")
    logger.info("Source files have been moved (not copied) to save storage space.")
    # Clean up empty directories in source
    logger.info("Cleaning up empty source directories...")
    for subset in ("LA", "PA", "DF"):
        subset_dir = ASVSPOOF_BASE / subset
        if subset_dir.exists():
            try:
                # Remove empty eval folders
                for eval_dir in subset_dir.glob(f"ASVspoof2021_{subset}_eval*"):
                    if eval_dir.is_dir():
                        shutil.rmtree(eval_dir)
                        logger.info(f"Cleaned: {eval_dir}")
            except Exception as e:
                logger.warning(f"Could not clean {subset_dir}: {e}")


if __name__ == "__main__":
    main()
