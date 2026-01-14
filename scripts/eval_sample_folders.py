"""
Evaluate all sample audio and image files in samples/real_* and samples/fake_* using
pretrained checkpoints. Prints per-file predictions and overall accuracy.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import torch

# Ensure project root on path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detect import load_config
from src.audio.pipeline import load_audio_model, infer_audio
from src.image.pipeline import load_image_model, infer_image


AUDIO_EXTS = (".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".wma")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def _gather(dir_path: Path, exts: tuple[str, ...]):
    files = []
    if dir_path.exists():
        for ext in exts:
            files.extend(dir_path.glob(f"*{ext}"))
    return sorted(files)


def evaluate_audio(cfg, device: torch.device, ckpt_dir: str):
    pairs = []
    for label_dir, label in (("real_audio", 0), ("fake_audio", 1)):
        pairs.extend((p, label) for p in _gather(Path("samples") / label_dir, AUDIO_EXTS))
    if not pairs:
        return None

    model = load_audio_model(cfg, device, ckpt_dir)
    thresh = float(cfg.get("audio", {}).get("threshold", 0.5))
    correct = 0
    rows = []
    for path, label in pairs:
        score = infer_audio(model, str(path), cfg, device)
        pred = 1 if score >= thresh else 0
        correct += int(pred == label)
        rows.append((path, score, pred, label))
    accuracy = correct / len(rows)
    return accuracy, rows


def evaluate_image(cfg, device: torch.device, ckpt_dir: str):
    pairs = []
    for label_dir, label in (("real_images", 0), ("fake_images", 1)):
        pairs.extend((p, label) for p in _gather(Path("samples") / label_dir, IMAGE_EXTS))
    if not pairs:
        return None

    model = load_image_model(cfg, device, ckpt_dir)
    thresh = float(cfg.get("image", {}).get("threshold", 0.5))
    correct = 0
    rows = []
    for path, label in pairs:
        score = infer_image(model, str(path), cfg, device)
        pred = 1 if score >= thresh else 0
        correct += int(pred == label)
        rows.append((path, score, pred, label))
    accuracy = correct / len(rows)
    return accuracy, rows


def main():
    parser = argparse.ArgumentParser(description="Evaluate samples/fake_* and samples/real_* with pretrained models")
    parser.add_argument("--config", default="configs/hparams.yaml", help="Path to config YAML")
    parser.add_argument("--checkpoint_dir", default="checkpoints", help="Directory containing checkpoints")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="cpu or cuda")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device)
    print(f"Using device: {device}")

    audio_eval = evaluate_audio(cfg, device, args.checkpoint_dir)
    if audio_eval is None:
        print("No audio samples found.")
    else:
        acc, rows = audio_eval
        print(f"\nAudio accuracy: {acc:.3f} ({len(rows)} samples)")
        for path, score, pred, label in rows:
            print(f"  {path}: score={score:.3f} pred={'FAKE' if pred else 'REAL'} label={'FAKE' if label else 'REAL'}")

    image_eval = evaluate_image(cfg, device, args.checkpoint_dir)
    if image_eval is None:
        print("No image samples found.")
    else:
        acc, rows = image_eval
        print(f"\nImage accuracy: {acc:.3f} ({len(rows)} samples)")
        for path, score, pred, label in rows:
            print(f"  {path}: score={score:.3f} pred={'FAKE' if pred else 'REAL'} label={'FAKE' if label else 'REAL'}")


if __name__ == "__main__":
    main()
