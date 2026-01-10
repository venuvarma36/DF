"""
Generate architecture diagram and classification report plots for the DeepFake project.

Outputs (default: reports/):
- architecture.png: high-level dataflow diagram.
- <modality>_confusion.png: confusion matrix heatmap.
- <modality>_prf.png: precision/recall/F1 bars per class.
- <modality>_roc.png: ROC curve (if both classes present).
- <modality>_report.txt: text classification report.

Labeling uses filename heuristic: names containing "fake" => FAKE (1); else REAL (0).

Run:
    python scripts/generate_reports.py --outdir reports
"""

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

from detect import load_config
from src.audio.pipeline import load_audio_model, infer_audio
from src.image.pipeline import load_image_model, infer_image


def label_from_name(path: str) -> int:
    name = os.path.basename(path).lower()
    return 1 if "fake" in name else 0


def gather_files(root: Path, exts: Tuple[str, ...]) -> List[Path]:
    files = [p for p in root.iterdir() if p.suffix.lower() in exts]
    files.sort()
    return files


def ensure_outdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def evaluate_audio(cfg, device, ckpt_dir: Path, sample_dir: Path):
    files = gather_files(sample_dir, (".wav",))
    if not files:
        return None
    model = load_audio_model(cfg, device, str(ckpt_dir))
    thresh = float(cfg.get("audio", {}).get("threshold", 0.5))
    labels, preds, scores, paths = [], [], [], []
    for path in files:
        label = label_from_name(str(path))
        score = float(infer_audio(model, str(path), cfg, device))
        pred = 1 if score >= thresh else 0
        labels.append(label)
        preds.append(pred)
        scores.append(score)
        paths.append(str(path))
    return {
        "labels": labels,
        "preds": preds,
        "scores": scores,
        "paths": paths,
        "threshold": thresh,
    }


def evaluate_image(cfg, device, ckpt_dir: Path, sample_dir: Path):
    files = gather_files(sample_dir, (".jpg", ".jpeg", ".png"))
    if not files:
        return None
    model = load_image_model(cfg, device, str(ckpt_dir))
    thresh = float(cfg.get("image", {}).get("threshold", 0.5))
    labels, preds, scores, paths = [], [], [], []
    for path in files:
        label = label_from_name(str(path))
        score = float(infer_image(model, str(path), cfg, device))
        pred = 1 if score >= thresh else 0
        labels.append(label)
        preds.append(pred)
        scores.append(score)
        paths.append(str(path))
    return {
        "labels": labels,
        "preds": preds,
        "scores": scores,
        "paths": paths,
        "threshold": thresh,
    }


def plot_confusion(cm: np.ndarray, classes: List[str], title: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_prf(report: dict, classes: List[str], title: str, out_path: Path):
    metrics = ["precision", "recall", "f1-score"]
    x = np.arange(len(classes))
    width = 0.25
    fig, ax = plt.subplots(figsize=(6, 4))
    for idx, m in enumerate(metrics):
        vals = [report.get(cls, {}).get(m, 0) for cls in classes]
        ax.bar(x + idx * width, vals, width, label=m.title())
    ax.set_xticks(x + width)
    ax.set_xticklabels(classes)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_roc_curve(labels: List[int], scores: List[float], title: str, out_path: Path):
    if len(set(labels)) < 2:
        return
    fpr, tpr, _ = roc_curve(labels, scores)
    auc = roc_auc_score(labels, scores)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"ROC AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_text_report(report: str, out_path: Path):
    out_path.write_text(report)


def draw_architecture(out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")

    def box(x, y, w, h, text):
        rect = plt.Rectangle((x, y), w, h, edgecolor="black", facecolor="#e8f1ff", linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)

    box(0.5, 1.2, 1.0, 0.5, "Input")
    box(0.0, 0.5, 1.0, 0.5, "Audio Pipeline\nSpecRNet-Lite")
    box(1.0, 0.5, 1.0, 0.5, "Image Pipeline\nSigLIP (HF)")
    box(0.0, -0.1, 1.0, 0.5, "Calibrated Audio Score")
    box(1.0, -0.1, 1.0, 0.5, "Calibrated Image Score")
    box(0.5, -0.8, 1.0, 0.5, "Fusion\nalpha·audio + beta·image")
    box(0.5, -1.5, 1.0, 0.5, "Final Decision")

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=1.5))

    arrow(1.0, 1.2, 0.5, 1.0)
    arrow(1.0, 1.2, 1.5, 1.0)
    arrow(0.5, 0.5, 0.5, 0.4)
    arrow(1.5, 0.5, 1.5, 0.4)
    arrow(0.5, -0.1, 1.0, -0.35)
    arrow(1.5, -0.1, 1.0, -0.35)
    arrow(1.0, -0.8, 1.0, -1.0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def process_modality(name: str, result: dict, outdir: Path):
    labels = result["labels"]
    preds = result["preds"]
    scores = result["scores"]
    classes = ["REAL", "FAKE"]

    cm = confusion_matrix(labels, preds, labels=[0, 1])
    plot_confusion(cm, classes, f"{name} Confusion Matrix", outdir / f"{name.lower()}_confusion.png")

    report_dict = classification_report(labels, preds, target_names=classes, output_dict=True, digits=3)
    report_text = classification_report(labels, preds, target_names=classes, digits=3)
    save_text_report(report_text, outdir / f"{name.lower()}_report.txt")
    plot_prf(report_dict, classes, f"{name} Precision/Recall/F1", outdir / f"{name.lower()}_prf.png")

    plot_roc_curve(labels, scores, f"{name} ROC", outdir / f"{name.lower()}_roc.png")



def main():
    parser = argparse.ArgumentParser(description="Generate architecture diagram and classification report plots.")
    parser.add_argument("--config", default="configs/hparams.yaml", help="Path to config file")
    parser.add_argument("--checkpoint_dir", default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--samples", default="samples", help="Samples directory")
    parser.add_argument("--outdir", default="reports", help="Output directory for PNGs and reports")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outdir = ensure_outdir(Path(args.outdir))
    ckpt_dir = Path(args.checkpoint_dir)
    sample_dir = Path(args.samples)

    print(f"Using device: {device}")
    print(f"Saving reports to: {outdir}")

    draw_architecture(outdir / "architecture.png")
    print("Saved architecture diagram.")

    audio_res = evaluate_audio(cfg, device, ckpt_dir, sample_dir)
    if audio_res:
        process_modality("Audio", audio_res, outdir)
        print(f"Audio accuracy: {np.mean(np.array(audio_res['preds']) == np.array(audio_res['labels'])):.3f}")
    else:
        print("No audio samples found.")

    image_res = evaluate_image(cfg, device, ckpt_dir, sample_dir)
    if image_res:
        process_modality("Image", image_res, outdir)
        print(f"Image accuracy: {np.mean(np.array(image_res['preds']) == np.array(image_res['labels'])):.3f}")
    else:
        print("No image samples found.")


if __name__ == "__main__":
    main()
