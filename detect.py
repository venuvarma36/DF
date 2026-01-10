import argparse
import os
import time
import torch
import yaml
from rich import print
from src.utils.cli_helpers import detect_modality
from src.audio.pipeline import load_audio_model, infer_audio
from src.image.pipeline import load_image_model, infer_image
from src.fusion.fusion import fuse_scores
from src.utils.metrics import format_inference_report


def load_config(path: str = "configs/hparams.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="DeepFake Detector (Audio + Image)")
    parser.add_argument("--audio", type=str, default=None, help="Path to audio file (.wav)")
    parser.add_argument("--image", type=str, default=None, help="Path to image file (.jpg/.png)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--config", type=str, default="configs/hparams.yaml")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp32", "fp16"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device)
    amp_dtype = torch.float16 if args.precision == "fp16" and device.type == "cuda" else torch.float32

    audio_thresh = float(cfg.get("audio", {}).get("threshold", 0.5))
    image_thresh = float(cfg.get("image", {}).get("threshold", 0.5))

    audio_path = args.audio
    image_path = args.image

    if not audio_path and not image_path:
        # attempt auto-detect from extension in a single input
        raise SystemExit("Provide --audio or --image path (or both).")

    # Prepare models lazily
    audio_model = None
    image_model = None

    reports = []

    if audio_path:
        if not os.path.exists(audio_path):
            raise SystemExit(f"Audio file not found: {audio_path}")
        audio_model = load_audio_model(cfg, device, args.checkpoint_dir)
        start = time.time()
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type=="cuda")):
            audio_score = infer_audio(audio_model, audio_path, cfg, device)
        latency_ms = (time.time() - start) * 1000
        reports.append(format_inference_report(
            modality="Audio", score=float(audio_score), latency_ms=latency_ms, threshold=audio_thresh
        ))

    if image_path:
        if not os.path.exists(image_path):
            raise SystemExit(f"Image file not found: {image_path}")
        image_model = load_image_model(cfg, device, args.checkpoint_dir)
        start = time.time()
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type=="cuda")):
            image_score = infer_image(image_model, image_path, cfg, device)
        latency_ms = (time.time() - start) * 1000
        reports.append(format_inference_report(
            modality="Image", score=float(image_score), latency_ms=latency_ms, threshold=image_thresh
        ))

    # Print individual modality results
    for r in reports:
        print(r)

    # Fusion if both present
    if audio_path and image_path:
        fused_score = fuse_scores(audio_score, image_score, cfg["fusion"]) if (audio_model and image_model) else None
        if fused_score is not None:
            pred = "FAKE" if fused_score >= 0.5 else "REAL"
            print({
                "Input Type": "Multimodal",
                "Prediction": pred,
                "Confidence": round(float(fused_score), 3)
            })


if __name__ == "__main__":
    main()
