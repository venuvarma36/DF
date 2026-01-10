"""
============================================================================
OPTIMIZED AUDIO TRAINING FOR RTX 2050 (4GB VRAM)
============================================================================

Configuration:
- Model: SpecRNet-Lite
- Batch size: 64 (train), 128 (val)
- Gradient accumulation: 1 step
- Gradient checkpointing: ENABLED (memory optimization)
- Mixed precision: FP16
- Feature caching: DISK (avoid recomputation)
- Epochs: 20 (2 warmup + 12 main + 6 finetune)
- Time target: ≤5 hours

Dataset Policy:
- Use 60k-80k audio clips max
- Keep all ASVspoof 2021
- Drop 80-90% of synthetic data

Outputs:
- accuracy, precision, recall, F1, ROC-AUC
- FAR, FRR, EER
- Inference latency (ms)
- Peak GPU memory (MB)
"""

from __future__ import annotations
import os
import yaml
import torch
import torchaudio
import pickle
import random
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from src.audio.pipeline import extract_features
from src.training.loops_optimized import train_loop_optimized


class BinaryAudioFolder(Dataset):
    """
    Load audio files from data/audio/{train,val,test}/{real,fake}
    
    With optional disk caching of mel-spectrogram features
    """
    
    def __init__(self, root: str, cfg: dict, split: str = 'train'):
        self.cfg = cfg
        self.split = split
        base = os.path.join(root, 'audio', split)
        self.items = []
        self.failed = set()
        
        for label, sub in [(0, 'real'), (1, 'fake')]:
            d = os.path.join(base, sub)
            if not os.path.isdir(d):
                continue
            for fn in os.listdir(d):
                if fn.lower().endswith(('.wav', '.flac', '.mp3', '.m4a')):
                    self.items.append((os.path.join(d, fn), label))
        
        # Optional downsampling and synthetic drop
        at_cfg = cfg.get('audio_training', {})
        max_samples = at_cfg.get('max_samples', 0)
        drop_synth_ratio = at_cfg.get('drop_synthetic_ratio', 0.0)
        random.seed(cfg.get('seed', 42))

        if drop_synth_ratio > 0:
            real = [(p, l) for p, l in self.items if l == 0]
            fake = [(p, l) for p, l in self.items if l == 1]
            keep_fake = int(len(fake) * (1.0 - drop_synth_ratio))
            fake = random.sample(fake, max(1, keep_fake)) if fake else []
            self.items = real + fake
            random.shuffle(self.items)

        if max_samples and len(self.items) > max_samples:
            self.items = random.sample(self.items, max_samples)

        # Setup feature cache if enabled
        self.enable_cache = at_cfg.get('enable_feature_caching', False)
        self.cache_dir = None
        
        if self.enable_cache:
            self.cache_dir = os.path.join(
                cfg.get('paths', {}).get('audio_cache', 'checkpoints/audio_cache'),
                split
            )
            os.makedirs(self.cache_dir, exist_ok=True)
            print(f"  Feature caching: ENABLED")
            print(f"  Cache dir: {self.cache_dir}")
        
        print(f"  {split.upper()}: {len(self.items)} audio files")

    def __len__(self):
        return len(self.items)
    
    def _get_cache_path(self, file_path):
        """Get cache file path for a given audio file"""
        file_hash = str(hash(file_path))
        return os.path.join(self.cache_dir, f"{file_hash}.pkl")
    
    def _load_or_compute_features(self, audio_path):
        """Load features from cache or compute them"""
        if audio_path in self.failed:
            return None

        if self.enable_cache:
            cache_path = self._get_cache_path(audio_path)
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
        
        # Compute features
        try:
            import soundfile as sf
            wav_np, sr = sf.read(audio_path, dtype='float32')
            if wav_np.size == 0:
                raise ValueError("empty audio")
            if len(wav_np.shape) > 1:  # stereo to mono
                wav_np = wav_np.mean(axis=1)
            wav = torch.from_numpy(wav_np).unsqueeze(0)  # [1, T]
        except Exception as e:
            # Fallback: try torchaudio backend before giving up
            try:
                wav, sr = torchaudio.load(audio_path)
                if wav.shape[0] > 1:
                    wav = wav.mean(dim=0, keepdim=True)
            except Exception as e2:
                if audio_path not in self.failed:
                    print(f"Warning: Failed to load {audio_path}: {e} | fallback: {e2}")
                    self.failed.add(audio_path)
                return None
            wav = wav.to(torch.float32)
        
        target_sr = self.cfg['audio_training'].get('sample_rate', 16000)
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
        
        wav = wav.to(torch.float32)  # ensure float32
        spec = extract_features(wav, self.cfg)  # [3, F, T] for 3 resolutions
        
        # Cache if enabled
        if self.enable_cache:
            cache_path = self._get_cache_path(audio_path)
            with open(cache_path, 'wb') as f:
                pickle.dump(spec, f)
        
        return spec

    def __getitem__(self, idx):
        path, label = self.items[idx]
        spec = self._load_or_compute_features(path)
        if spec is None:
            return None
        return spec.squeeze(0), torch.tensor(label, dtype=torch.long)


def collate_audio(batch):
    """Collate that drops failed samples and pads time dim to max in batch."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    specs, labels = zip(*batch)
    # specs: list of [3, F, T]
    max_T = max(s.shape[-1] for s in specs)
    padded = []
    for s in specs:
        pad_T = max_T - s.shape[-1]
        if pad_T > 0:
            s = torch.nn.functional.pad(s, (0, pad_T))
        padded.append(s)
    return torch.stack(padded, dim=0), torch.stack(labels, dim=0)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Optimized Audio Training for RTX 2050")
    parser.add_argument('--config', type=str, default='configs/optimized_rtx2050.yaml', help='Config file')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    parser.add_argument('--data_root', type=str, default='data', help='Data root directory')
    parser.add_argument('--checkpoints', type=str, default='checkpoints', help='Checkpoints directory')
    parser.add_argument('--cache-features', action='store_true', help='Enable disk-based feature caching')
    args = parser.parse_args()

    print("\n" + "="*80)
    print("🎙️  OPTIMIZED AUDIO TRAINING - RTX 2050 (4GB VRAM)")
    print("="*80)
    
    # Load config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Enable feature caching if requested
    if args.cache_features:
        cfg['audio_training']['enable_feature_caching'] = True
    
    print(f"\n📋 Configuration: {args.config}")
    print(f"🖥️  Device: {args.device}")
    print(f"📁 Data root: {args.data_root}")
    
    # Setup device
    device = torch.device(args.device)
    if device.type == "cuda":
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Load datasets with feature caching
    print(f"\n📥 Loading datasets (with {'cached' if args.cache_features else 'on-the-fly'} features)...")
    train_ds = BinaryAudioFolder(args.data_root, cfg, split='train')
    val_ds = BinaryAudioFolder(args.data_root, cfg, split='val')
    
    # Build model
    print(f"\n🏗️  Building model...")
    from src.audio.model_specRNet import SpecRNetLite
    
    feat_cfg = cfg['audio_training']
    model = SpecRNetLite(
        in_branches=len(feat_cfg.get('multi_res_ffts', [256, 512, 1024])),
        channels=feat_cfg.get('model_params', {}).get('channels', [16, 32, 64]),
        gru_hidden=feat_cfg.get('model_params', {}).get('gru_hidden', 128),
        embed_dim=feat_cfg.get('model_params', {}).get('embed_dim', 128),
        dropout=feat_cfg.get('model_params', {}).get('dropout', 0.2)
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {num_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Setup checkpoint path
    os.makedirs(args.checkpoints, exist_ok=True)
    ckpt_path = os.path.join(args.checkpoints, 'audio_best.pt')
    
    # Train
    print(f"\n🚀 Starting training loop...")
    best_eer, history = train_loop_optimized(
        model, 
        train_ds, 
        val_ds, 
        cfg, 
        device, 
        ckpt_path, 
        task='audio',
        collate_fn=collate_audio
    )
    
    print(f"\n✅ AUDIO TRAINING COMPLETE")
    print(f"   Best EER: {best_eer:.4f}")
    print(f"   Checkpoint: {ckpt_path}")
    
    # Cleanup cache if created
    if args.cache_features:
        print(f"\n💾 Feature cache directory: {cfg.get('paths', {}).get('audio_cache', 'checkpoints/audio_cache')}")


if __name__ == '__main__':
    main()
