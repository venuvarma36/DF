import os
import sys
from pathlib import Path
import torch

# Ensure project root on path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detect import load_config
from src.audio.pipeline import load_audio_model, infer_audio
from src.image.pipeline import load_image_model, infer_image


def _label_from_name(path: str) -> int:
    name = os.path.basename(path).lower()
    return 1 if "fake" in name else 0


def evaluate_audio(cfg, device, ckpt_dir: str = "checkpoints"):
    files = [os.path.join("samples", f) for f in os.listdir("samples") if f.lower().endswith(".wav")]
    if not files:
        return None
    files.sort()
    model = load_audio_model(cfg, device, ckpt_dir)
    thresh = float(cfg.get("audio", {}).get("threshold", 0.5))
    correct = 0
    results = []
    for path in files:
        label = _label_from_name(path)
        score = infer_audio(model, path, cfg, device)
        pred = 1 if score >= thresh else 0
        correct += int(pred == label)
        results.append((path, score, pred, label))
    accuracy = correct / len(files)
    return accuracy, results


def evaluate_image(cfg, device, ckpt_dir: str = "checkpoints"):
    files = [os.path.join("samples", f) for f in os.listdir("samples") if f.lower().endswith((".jpg", ".png"))]
    if not files:
        return None
    files.sort()
    model = load_image_model(cfg, device, ckpt_dir)
    thresh = float(cfg.get("image", {}).get("threshold", 0.5))
    correct = 0
    results = []
    for path in files:
        label = _label_from_name(path)
        score = infer_image(model, path, cfg, device)
        pred = 1 if score >= thresh else 0
        correct += int(pred == label)
        results.append((path, score, pred, label))
    accuracy = correct / len(files)
    return accuracy, results


def main():
    cfg = load_config("configs/hparams.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    audio_eval = evaluate_audio(cfg, device)
    if audio_eval is not None:
        acc, rows = audio_eval
        print(f"\nAudio accuracy: {acc:.3f} ({len(rows)} samples)")
        for path, score, pred, label in rows:
            print(f"  {path}: score={score:.3f} pred={'FAKE' if pred else 'REAL'} label={'FAKE' if label else 'REAL'}")
    else:
        print("No audio samples found.")

    image_eval = evaluate_image(cfg, device)
    if image_eval is not None:
        acc, rows = image_eval
        print(f"\nImage accuracy: {acc:.3f} ({len(rows)} samples)")
        for path, score, pred, label in rows:
            print(f"  {path}: score={score:.3f} pred={'FAKE' if pred else 'REAL'} label={'FAKE' if label else 'REAL'}")
    else:
        print("No image samples found.")


if __name__ == "__main__":
    main()
