from __future__ import annotations
import time
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns


def compute_classification_metrics(y_true, y_pred, y_score=None):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_score) if y_score is not None and len(np.unique(y_true)) == 2 else float('nan')
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
    }


def compute_security_metrics(y_true, y_score):
    # y_true: 1 = FAKE (positive), 0 = REAL (negative)
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()

    # Drop NaNs/Infs in scores to avoid NaN slices
    mask = np.isfinite(y_score)
    y_true = y_true[mask]
    y_score = y_score[mask]

    # If we do not have both classes present, return NaN metrics safely
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return {
            "FAR_curve": np.array([]),
            "FRR_curve": np.array([]),
            "thresholds": np.array([]),
            "EER": float('nan'),
        }

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    far = fpr  # False Acceptance Rate = FP / (FP + TN)
    frr = 1 - tpr  # False Rejection Rate = FN / (TP + FN)

    # Handle empty or NaN-only arrays safely
    if thresholds.size == 0 or far.size == 0 or frr.size == 0:
        return {
            "FAR_curve": far,
            "FRR_curve": frr,
            "thresholds": thresholds,
            "EER": float('nan'),
        }

    diff = np.abs(far - frr)
    if diff.size == 0 or np.all(np.isnan(diff)):
        return {
            "FAR_curve": far,
            "FRR_curve": frr,
            "thresholds": thresholds,
            "EER": float('nan'),
        }

    idx = np.nanargmin(diff)
    eer = (far[idx] + frr[idx]) / 2.0
    return {
        "FAR_curve": far,
        "FRR_curve": frr,
        "thresholds": thresholds,
        "EER": float(eer),
    }


def latency_ms(fn, *args, **kwargs):
    t0 = time.time()
    out = fn(*args, **kwargs)
    ms = (time.time() - t0) * 1000
    return out, ms


def gpu_memory_mb():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / (1024**2)
    return 0.0


def param_count(model: torch.nn.Module):
    return sum(p.numel() for p in model.parameters())


def plot_confusion(y_true, y_pred, normalize=False, title="Confusion Matrix", save_path=None):
    cm = confusion_matrix(y_true, y_pred, normalize='true' if normalize else None)
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_roc(y_true, y_score, save_path=None):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    plt.figure(figsize=(4, 4))
    plt.plot(fpr, tpr, label=f'AUC={auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
    plt.title('ROC Curve')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_pr(y_true, y_score, save_path=None):
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    plt.figure(figsize=(4, 4))
    plt.plot(rec, prec)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def format_inference_report(modality: str, score: float, latency_ms: float, threshold: float = 0.5):
    pred = "FAKE" if score >= threshold else "REAL"
    # In a single-sample inference, we cannot derive FAR/FRR; present placeholders
    if modality.lower() == "audio":
        return {
            "Input Type": "Audio",
            "Prediction": pred,
            "Confidence": round(score, 3),
            "FAR": None,
            "FRR": None,
            "Inference Time": f"{latency_ms:.1f} ms",
        }
    else:
        # Image: show precision/recall placeholders as None at inference-only
        return {
            "Input Type": "Image",
            "Prediction": pred,
            "Confidence": round(score, 3),
            "Precision": None,
            "Recall": None,
            "Inference Time": f"{latency_ms:.1f} ms",
        }


def plot_training_curves(history: dict, save_path: str = None):
    # history keys: 'epoch', 'train_loss', 'val_acc', 'val_eer'
    epochs = list(history.get('epoch', []))
    tl = list(history.get('train_loss', []))
    va = list(history.get('val_acc', []))
    ve = list(history.get('val_eer', []))

    if not epochs:
        return

    # Align lengths defensively so plotting never fails when val metrics are missing
    def _trim(x, ref_len):
        return x[:min(len(x), ref_len)]

    tl = _trim(tl, len(epochs))
    va = _trim(va, len(epochs))
    ve = _trim(ve, len(epochs))
    epochs_tl = epochs[:len(tl)]

    fig, ax1 = plt.subplots(figsize=(6,4))
    color1 = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss/Accuracy', color=color1)
    if tl:
        ax1.plot(epochs_tl, tl, label='Train Loss', color='tab:blue')
    if va:
        ax1.plot(epochs[:len(va)], va, label='Val Acc', color='tab:green')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('EER', color=color2)
    if ve:
        ax2.plot(epochs[:len(ve)], ve, label='Val EER', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()
