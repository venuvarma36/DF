"""
Train audio model with balanced dataset (half the data to balance classes)
"""

import os
import sys
import yaml
import torch
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__) + "/..")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.training.train_audio_optimized import main as train_audio_main

def main():
    print("=" * 80)
    print("BALANCED AUDIO MODEL TRAINING")
    print("=" * 80)
    
    # Load config
    cfg_path = "configs/optimized_rtx2050.yaml"
    cfg = yaml.safe_load(open(cfg_path, 'r'))
    
    # Verify dataset
    print("\n📊 Dataset Distribution:")
    train_real = len(list(Path("data/audio/train/real").glob("*.wav")))
    train_fake = len(list(Path("data/audio/train/fake").glob("*.wav")))
    val_real = len(list(Path("data/audio/val/real").glob("*.wav")))
    val_fake = len(list(Path("data/audio/val/fake").glob("*.wav")))
    
    print(f"  Train: Real={train_real}, Fake={train_fake}, Total={train_real+train_fake}")
    print(f"  Val:   Real={val_real}, Fake={val_fake}, Total={val_real+val_fake}")
    print(f"  Ratio: 1:{train_fake/train_real:.1f} (Fake:Real)")
    
    # Balance by limiting fake samples to match real samples
    balance_ratio = train_fake / train_real
    if balance_ratio > 2:
        print(f"\n⚠️  Dataset is imbalanced ({balance_ratio:.1f}:1 ratio)")
        print(f"💡 Recommendation: Limit fake samples to ~{train_real * 2} for better balance")
        max_fake_samples = min(train_fake, train_real * 2)
        print(f"   Using max {max_fake_samples} fake samples for training")
        
        # Update config to limit samples
        if 'audio' not in cfg:
            cfg['audio'] = {}
        if 'data' not in cfg['audio']:
            cfg['audio']['data'] = {}
        cfg['audio']['data']['max_samples_per_class'] = int(max_fake_samples)
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Device: {device}")
    
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    print("\n" + "=" * 80)
    print("STARTING AUDIO TRAINING")
    print("=" * 80)
    
    start_time = time.time()
    
    try:
        # Run training
        train_audio_main()
        
        elapsed_time = time.time() - start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        
        print("\n" + "=" * 80)
        print("✅ TRAINING COMPLETE!")
        print("=" * 80)
        print(f"⏱️  Training time: {hours}h {minutes}m")
        print(f"📁 Checkpoint saved: checkpoints/audio_best.pt")
        print(f"📊 Training curves: logs/audio_training_curves.png")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Test the trained model
    print("\n" + "=" * 80)
    print("TESTING TRAINED MODEL")
    print("=" * 80)
    
    try:
        from scripts.test_audio_model import main as test_main
        test_main()
    except Exception as e:
        print(f"⚠️  Could not run automatic testing: {e}")
        print("   Run manually with: python scripts/test_audio_model.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
