from __future__ import annotations
import os
import time
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from src.utils.metrics import (
    compute_classification_metrics,
    compute_security_metrics,
    plot_confusion,
    plot_roc,
    plot_pr,
    gpu_memory_mb,
    param_count,
)
from src.audio.pipeline import extract_features, load_audio_model, preprocess_audio
from src.image.pipeline import load_image_model, preprocess_image


class BinaryAudioFolder(Dataset):
    def __init__(self, root: str, cfg: dict, split: str):
        self.cfg = cfg
        base = os.path.join(root, 'audio', split)
        self.items = []
        for label, sub in [(0, 'real'), (1, 'fake')]:
            d = os.path.join(base, sub)
            if not os.path.isdir(d):
                continue
            for fn in os.listdir(d):
                if fn.lower().endswith(('.wav', '.flac', '.mp3', '.m4a')):
                    self.items.append((os.path.join(d, fn), label))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        wav = preprocess_audio(path, self.cfg, device=torch.device('cpu'))
        spec = extract_features(wav, self.cfg).squeeze(0)
        return spec, torch.tensor(label, dtype=torch.long), path


class BinaryImageFolder(Dataset):
    def __init__(self, root: str, cfg: dict, split: str):
        self.cfg = cfg
        base = os.path.join(root, 'image', split)
        self.items = []
        for label, sub in [(0, 'real'), (1, 'fake')]:
            d = os.path.join(base, sub)
            if not os.path.isdir(d):
                continue
            for fn in os.listdir(d):
                if fn.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    self.items.append((os.path.join(d, fn), label))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = preprocess_image(path, self.cfg['image']['params'].get('img_size', 224)).squeeze(0)
        return img, torch.tensor(label, dtype=torch.long), path


def evaluate(task: str, split: str, cfg: dict, device: torch.device, ckpt_dir: str, logs_dir: str):
    os.makedirs(logs_dir, exist_ok=True)
    if task == 'audio':
        ds = BinaryAudioFolder(cfg['paths']['data_root'], cfg, split)
        dl = DataLoader(ds, batch_size=1, shuffle=False)
        model = load_audio_model(cfg, device, ckpt_dir)
    else:
        ds = BinaryImageFolder(cfg['paths']['data_root'], cfg, split)
        dl = DataLoader(ds, batch_size=1, shuffle=False)
        model = load_image_model(cfg, device, ckpt_dir)

    ys, ps, preds = [], [], []
    latencies = []
    with torch.no_grad():
        for x, y, _ in dl:
            x = x.to(device)
            y = y.to(device)
            t0 = time.time()
            out = model(x)
            latencies.append((time.time() - t0) * 1000)
            score = out.detach().cpu().numpy().reshape(-1)
            ps.append(score)
            ys.append(y.detach().cpu().numpy().reshape(-1))
            preds.append((score >= 0.5).astype(int))

    ys = np.concatenate(ys)
    ps = np.concatenate(ps)
    preds = np.concatenate(preds)

    cls = compute_classification_metrics(ys, preds, ps)
    sec = compute_security_metrics(ys, ps)

    # Plots
    plot_confusion(ys, preds, normalize=False, title=f'{task} Confusion (raw)', save_path=os.path.join(logs_dir, f'{task}_{split}_confusion_raw.png'))
    plot_confusion(ys, preds, normalize=True, title=f'{task} Confusion (norm)', save_path=os.path.join(logs_dir, f'{task}_{split}_confusion_norm.png'))
    plot_roc(ys, ps, save_path=os.path.join(logs_dir, f'{task}_{split}_roc.png'))
    plot_pr(ys, ps, save_path=os.path.join(logs_dir, f'{task}_{split}_pr.png'))

    report = {
        'split': split,
        'samples': int(len(ds)),
        'accuracy': float(cls['accuracy']),
        'precision': float(cls['precision']),
        'recall': float(cls['recall']),
        'f1': float(cls['f1']),
        'roc_auc': float(cls['roc_auc']),
        'FAR_at_thresholds': sec['FAR_curve'][:5].tolist(),
        'FRR_at_thresholds': sec['FRR_curve'][:5].tolist(),
        'EER': float(sec['EER']),
        'latency_ms_mean': float(np.mean(latencies)),
        'latency_ms_p95': float(np.percentile(latencies, 95)),
        'gpu_mem_mb_peak': float(gpu_memory_mb()),
        'param_count': int(param_count(model)),
    }
    print(report)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/hparams.yaml')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--task', type=str, choices=['audio','image'], required=True)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--checkpoints', type=str, default='checkpoints')
    parser.add_argument('--logs', type=str, default='logs')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device)
    evaluate(args.task, args.split, cfg, device, args.checkpoints, args.logs)


if __name__ == '__main__':
    main()
