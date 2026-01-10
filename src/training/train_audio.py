from __future__ import annotations
import os
import yaml
import torch
from torch.utils.data import Dataset
from src.audio.pipeline import load_audio_model, extract_features
from src.training.loops import train_loop
import torchaudio


class BinaryAudioFolder(Dataset):
    def __init__(self, root: str, cfg: dict, split: str = 'train'):
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
        wav, sr = torchaudio.load(path)
        target_sr = self.cfg['audio']['feature']['sample_rate']
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
        wav = wav.mean(dim=0, keepdim=True).to(torch.float32)  # mono
        spec = extract_features(wav, self.cfg)  # [1, R, F, T]
        return spec.squeeze(0), torch.tensor(label, dtype=torch.long)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/hparams.yaml')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--checkpoints', type=str, default='checkpoints')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device)

    train_ds = BinaryAudioFolder(cfg['paths']['data_root'], cfg, split='train')
    val_ds = BinaryAudioFolder(cfg['paths']['data_root'], cfg, split='val')

    # Build model
    from src.audio.model_specRNet import SpecRNetLite
    feat_cfg = cfg['audio']['feature']
    model = SpecRNetLite(in_branches=len(feat_cfg.get('multi_res_ffts', [512])),
                         channels=cfg['audio']['params'].get('channels', [16,32,64]),
                         gru_hidden=cfg['audio']['params'].get('gru_hidden', 128),
                         embed_dim=cfg['audio']['params'].get('embed_dim', 128),
                         dropout=cfg['audio']['params'].get('dropout', 0.2))

    ckpt_path = os.path.join(cfg['paths']['checkpoints'], 'audio_best.pt')
    best_eer = train_loop(model, train_ds, val_ds, cfg, device, ckpt_path, task='audio')
    print({'best_val_eer': best_eer})


if __name__ == '__main__':
    main()
