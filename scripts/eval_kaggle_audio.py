"""
Evaluate the best Kaggle audio model: ROC-AUC, PR-AUC, confusion matrix, per-class metrics.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchaudio
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, classification_report
import json
import sys

# Ensure src is importable
sys.path.insert(0, r'C:\DeepFake_detection')

# Config (must match training)
CONFIG = {
    'sample_rate': 16000,
    'n_mels': 80,
    'n_fft': 400,
    'hop_length': 160,
    'duration_sec': 5.0,
    'batch_size': 64,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

DATA_PATH = Path(r'C:\DeepFake_detection\data\audio')
CHECKPOINT_PATH = Path(r'C:\DeepFake_detection\checkpoints\audio_kaggle_best.pt')
OUTPUT_PATH = Path(r'C:\DeepFake_detection\logs\eval_metrics.json')

# Model
from src.audio.model_specRNet import SpecRNetLite

class AudioDataset(Dataset):
    def __init__(self, split='test'):
        self.files = []
        self.labels = []
        for label, name in [(0, 'real'), (1, 'fake')]:
            for p in (DATA_PATH / split / name).glob('*.wav'):
                self.files.append(p)
                self.labels.append(label)
        print(f"{split}: {len(self.files)} files")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        label = self.labels[idx]
        wav, sr = torchaudio.load(str(path))
        if sr != CONFIG['sample_rate']:
            wav = torchaudio.transforms.Resample(sr, CONFIG['sample_rate'])(wav)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if wav.abs().max() > 0:
            wav = wav / wav.abs().max()
        target_len = int(CONFIG['duration_sec'] * CONFIG['sample_rate'])
        cur_len = wav.shape[1]
        if cur_len > target_len:
            wav = wav[:, :target_len]
        elif cur_len < target_len:
            wav = torch.cat([wav, torch.zeros(1, target_len - cur_len)], dim=1)
        return wav, label

def collate_fn(batch):
    waves, labels = zip(*batch)
    waves = torch.cat(waves, dim=0)  # they are already padded to same length
    labels = torch.tensor(labels, dtype=torch.float32)
    return waves, labels

def extract_features(waveform):
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=CONFIG['sample_rate'],
        n_mels=CONFIG['n_mels'],
        n_fft=CONFIG['n_fft'],
        hop_length=CONFIG['hop_length'],
        power=2.0,
        center=True,
        pad_mode='reflect',
    ).to(CONFIG['device'])(waveform)
    mel = torch.clamp(mel, min=1e-9)
    mel = torch.log(mel)
    if mel.dim() == 3:
        mel = mel.unsqueeze(1)
    return mel

def main():
    device = torch.device(CONFIG['device'])
    ds = AudioDataset('test')
    loader = DataLoader(ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0, collate_fn=collate_fn)

    model = SpecRNetLite(in_branches=1).to(device)
    state = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(loader, desc='Eval')
        for waves, labels in pbar:
            waves = waves.to(device)
            labels = labels.to(device)
            feats = extract_features(waves)
            probs = model(feats)
            all_probs.append(probs.detach().cpu())
            all_labels.append(labels.detach().cpu())
    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()

    auc = roc_auc_score(all_labels, all_probs)
    ap = average_precision_score(all_labels, all_probs)
    preds = (all_probs > 0.5).astype(int)
    cm = confusion_matrix(all_labels, preds)
    report = classification_report(all_labels, preds, output_dict=True, digits=4)

    results = {
        'roc_auc': float(auc),
        'average_precision': float(ap),
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved metrics to {OUTPUT_PATH}")
    print(f"ROC-AUC: {auc:.4f} | PR-AUC: {ap:.4f}")
    print('Confusion matrix:', cm)

if __name__ == '__main__':
    main()
