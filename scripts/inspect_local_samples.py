"""Inspect local sample wavs for duration, peak, and mel energy distribution."""
import torchaudio
import torch
from pathlib import Path

SAMPLES_DIR = Path(r"C:\DeepFake_detection\samples")
TARGET_SR = 16000
TARGET_LEN = TARGET_SR * 5

files = []
for label in ["real", "fake"]:
    files.extend([(label, p) for p in sorted(SAMPLES_DIR.glob(f"{label}_*.wav"))])

if not files:
    print("No sample wavs found.")
    raise SystemExit

print(f"Found {len(files)} files")

rows = []
for label, path in files:
    wav, sr = torchaudio.load(str(path))
    if sr != TARGET_SR:
        wav = torchaudio.transforms.Resample(sr, TARGET_SR)(wav)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    peak = wav.abs().max().item()
    dur = wav.shape[1] / TARGET_SR
    energy = wav.pow(2).mean().item()
    rows.append((label, path.name, dur, peak, energy))

print("label,file,duration_sec,peak,mean_energy")
for r in rows:
    print(f"{r[0]},{r[1]},{r[2]:.3f},{r[3]:.4f},{r[4]:.6f}")
