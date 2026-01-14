"""Generate metric plots (accuracy, confusion, PR/F1, ROC) for samples.

Run:
    python scripts/generate_metrics_diagrams.py --outdir diagrams

Uses filename heuristic: filenames containing "fake" -> FAKE else REAL.
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

DPI = 150


def label_from_name(path: str) -> int:
    return 1 if "fake" in os.path.basename(path).lower() else 0


def gather_files(root: Path, exts: Tuple[str, ...]) -> List[Path]:
    files = [p for p in root.iterdir() if p.suffix.lower() in exts]
    files.sort()
    return files


def ensure_outdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def evaluate(cfg, device, ckpt_dir: Path, sample_dir: Path, mode: str):
    if mode == "audio":
        files = gather_files(sample_dir, (".wav",))
        if not files:
            return None
        model = load_audio_model(cfg, device, str(ckpt_dir))
        thresh = float(cfg.get("audio", {}).get("threshold", 0.5))
        infer_fn = lambda p: float(infer_audio(model, p, cfg, device))
    else:
        files = gather_files(sample_dir, (".jpg", ".jpeg", ".png"))
        if not files:
            return None
        model = load_image_model(cfg, device, str(ckpt_dir))
        thresh = float(cfg.get("image", {}).get("threshold", 0.5))
        infer_fn = lambda p: float(infer_image(model, p, cfg, device))

    labels, preds, scores = [], [], []
    for path in files:
        label = label_from_name(str(path))
        score = infer_fn(str(path))
        pred = 1 if score >= thresh else 0
        labels.append(label)
        preds.append(pred)
        scores.append(score)
    return {
        "labels": labels,
        "preds": preds,
        "scores": scores,
        "accuracy": float(np.mean(np.array(preds) == np.array(labels))),
    }


def plot_confusion(cm: np.ndarray, classes: List[str], title: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(4, 3.6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ticks = np.arange(len(classes))
    ax.set_xticks(ticks); ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(ticks); ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_prf(report: dict, classes: List[str], title: str, out_path: Path):
    metrics = ["precision", "recall", "f1-score"]
    x = np.arange(len(classes))
    width = 0.25
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    for idx, m in enumerate(metrics):
        vals = [report.get(cls, {}).get(m, 0) for cls in classes]
        ax.bar(x + idx * width, vals, width, label=m.title())
    ax.set_xticks(x + width)
    ax.set_xticklabels(classes)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_roc(labels: List[int], scores: List[float], title: str, out_path: Path):
    if len(set(labels)) < 2:
        return
    fpr, tpr, _ = roc_curve(labels, scores)
    auc = roc_auc_score(labels, scores)
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.plot(fpr, tpr, label=f"ROC AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title(title)
    ax.legend(loc="lower right")
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_bar(results: dict, out_path: Path):
    names, accs = [], []
    for k, v in results.items():
        if v:
            names.append(k.capitalize())
            accs.append(v["accuracy"])
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ax.bar(names, accs, color=["#5b8ff9", "#5ad8a6"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    for i, v in enumerate(accs):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom")
    ax.set_title("Sample Accuracy")
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def process(modality: str, res: dict, outdir: Path):
    classes = ["REAL", "FAKE"]
    cm = confusion_matrix(res["labels"], res["preds"], labels=[0, 1])
    plot_confusion(cm, classes, f"{modality.capitalize()} Confusion", outdir / f"metrics_{modality}_confusion.png")
    report_dict = classification_report(res["labels"], res["preds"], target_names=classes, output_dict=True, digits=3)
    plot_prf(report_dict, classes, f"{modality.capitalize()} PR/F1", outdir / f"metrics_{modality}_prf.png")
    plot_roc(res["labels"], res["scores"], f"{modality.capitalize()} ROC", outdir / f"metrics_{modality}_roc.png")


def main():
    parser = argparse.ArgumentParser(description="Generate metric plots for samples.")
    parser.add_argument("--config", default="configs/hparams.yaml")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--samples", default="samples")
    parser.add_argument("--outdir", default="diagrams")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outdir = ensure_outdir(Path(args.outdir))
    ckpt_dir = Path(args.checkpoint_dir)
    sample_dir = Path(args.samples)

    audio_res = evaluate(cfg, device, ckpt_dir, sample_dir, "audio")
    image_res = evaluate(cfg, device, ckpt_dir, sample_dir, "image")

    results = {"audio": audio_res, "image": image_res}
    if audio_res:
        process("audio", audio_res, outdir)
    if image_res:
        process("image", image_res, outdir)
    if audio_res or image_res:
        plot_accuracy_bar(results, outdir / "metrics_accuracy.png")

    if not (audio_res or image_res):
        print("No samples found.")
    else:
        print(f"Saved metric plots to {outdir}")


if __name__ == "__main__":
    main()
