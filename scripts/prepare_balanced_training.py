"""
Balance audio dataset and train new model
Balances train/test/val splits by matching the minority class count
"""

import os
import sys
import random
import shutil
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def scan_dataset(data_dir):
    """Scan and count files in dataset"""
    print("\n" + "="*80)
    print("SCANNING DATASET")
    print("="*80)
    
    splits = ['train', 'val', 'test']
    classes = ['real', 'fake']
    
    counts = {}
    
    for split in splits:
        counts[split] = {}
        for cls in classes:
            path = data_dir / split / cls
            if path.exists():
                wav_files = list(path.glob("*.wav"))
                counts[split][cls] = len(wav_files)
                print(f"{split}/{cls}: {len(wav_files)} files")
            else:
                counts[split][cls] = 0
                print(f"{split}/{cls}: 0 files (directory not found)")
    
    return counts

def balance_dataset(data_dir, counts):
    """Balance dataset by limiting to minority class size"""
    print("\n" + "="*80)
    print("BALANCING DATASET")
    print("="*80)
    
    splits = ['train', 'val', 'test']
    balanced_counts = {}
    
    for split in splits:
        real_count = counts[split]['real']
        fake_count = counts[split]['fake']
        
        # Use minimum count for balance
        target_count = min(real_count, fake_count)
        
        print(f"\n{split.upper()}:")
        print(f"  Real: {real_count} files")
        print(f"  Fake: {fake_count} files")
        print(f"  Target (balanced): {target_count} files per class")
        
        balanced_counts[split] = {
            'real': real_count,
            'fake': fake_count,
            'target': target_count
        }
    
    return balanced_counts

def train_balanced_model(data_dir, balanced_counts):
    """Train model with balanced dataset"""
    print("\n" + "="*80)
    print("TRAINING NEW MODEL WITH BALANCED DATASET")
    print("="*80)
    
    # Calculate total samples
    total_train = balanced_counts['train']['target'] * 2
    total_val = balanced_counts['val']['target'] * 2
    total_test = balanced_counts['test']['target'] * 2
    
    print(f"\nBalanced dataset size:")
    print(f"  Train: {total_train} samples ({balanced_counts['train']['target']} per class)")
    print(f"  Val: {total_val} samples ({balanced_counts['val']['target']} per class)")
    print(f"  Test: {total_test} samples ({balanced_counts['test']['target']} per class)")
    
    # Create training script
    train_script = """
import os
import sys
import yaml
import torch
import random
import numpy as np
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.audio.model_specRNet import SpecRNet_Lite
from src.training.loops_optimized import train_audio_with_validation
from src.utils.metrics import evaluate_model

def get_balanced_samples(data_dir, split, target_count):
    \"\"\"Get balanced samples from dataset\"\"\"
    real_dir = data_dir / split / 'real'
    fake_dir = data_dir / split / 'fake'
    
    real_files = list(real_dir.glob('*.wav'))
    fake_files = list(fake_dir.glob('*.wav'))
    
    # Sample exactly target_count from each class
    real_samples = random.sample(real_files, min(target_count, len(real_files)))
    fake_samples = random.sample(fake_files, min(target_count, len(fake_files)))
    
    # Create combined list with labels
    samples = []
    for f in real_samples:
        samples.append((str(f), 0))  # 0 = real
    for f in fake_samples:
        samples.append((str(f), 1))  # 1 = fake
    
    random.shuffle(samples)
    return samples

def main():
    print("="*80)
    print("TRAINING BALANCED AUDIO DEEPFAKE DETECTION MODEL")
    print("="*80)
    
    # Load config
    cfg = yaml.safe_load(open('configs/hparams.yaml', 'r'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\\nDevice: {device}")
    
    # Set random seed
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Data directory
    data_dir = Path('data/audio')
    
    # Balanced counts from scan
    balanced_counts = {
        'train': {'target': %(train_target)d},
        'val': {'target': %(val_target)d},
        'test': {'target': %(test_target)d}
    }
    
    print(f"\\nBalanced dataset:")
    print(f"  Train: {balanced_counts['train']['target']*2} samples")
    print(f"  Val: {balanced_counts['val']['target']*2} samples")
    print(f"  Test: {balanced_counts['test']['target']*2} samples")
    
    # Get balanced samples
    train_samples = get_balanced_samples(data_dir, 'train', balanced_counts['train']['target'])
    val_samples = get_balanced_samples(data_dir, 'val', balanced_counts['val']['target'])
    test_samples = get_balanced_samples(data_dir, 'test', balanced_counts['test']['target'])
    
    print(f"\\nActual loaded samples:")
    print(f"  Train: {len(train_samples)}")
    print(f"  Val: {len(val_samples)}")
    print(f"  Test: {len(test_samples)}")
    
    # Initialize model
    print("\\nInitializing SpecRNet-Lite model...")
    model = SpecRNet_Lite(num_classes=1)
    model = model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training configuration
    cfg['training'] = {
        'epochs': 25,
        'batch_size': 64,
        'learning_rate': 0.0001,
        'warmup_epochs': 3,
        'early_stopping_patience': 5,
        'use_amp': True
    }
    
    print(f"\\nTraining configuration:")
    print(f"  Epochs: {cfg['training']['epochs']}")
    print(f"  Batch size: {cfg['training']['batch_size']}")
    print(f"  Learning rate: {cfg['training']['learning_rate']}")
    print(f"  Early stopping patience: {cfg['training']['early_stopping_patience']}")
    print(f"  Mixed precision: {cfg['training']['use_amp']}")
    
    # Create checkpoint directory
    checkpoint_dir = Path('checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)
    
    print("\\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)
    
    # Import training function
    from src.training.train_audio_optimized import train_audio_model
    
    # Train model
    best_model, history = train_audio_model(
        train_files=train_samples,
        val_files=val_samples,
        cfg=cfg,
        device=device,
        checkpoint_dir=str(checkpoint_dir),
        model_name='audio_balanced_new'
    )
    
    print("\\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"\\nBest model saved to: checkpoints/audio_balanced_new.pt")
    
    # Test on test set
    print("\\n" + "="*80)
    print("TESTING ON TEST SET")
    print("="*80)
    
    from src.audio.pipeline import load_audio_model, infer_audio
    
    model = load_audio_model(cfg, device, 'checkpoints', model_file='audio_balanced_new.pt')
    
    correct = 0
    real_correct = 0
    fake_correct = 0
    real_total = 0
    fake_total = 0
    
    for file_path, label in test_samples:
        score = infer_audio(model, file_path, cfg, device)
        prediction = 1 if score > 0.5 else 0
        
        if prediction == label:
            correct += 1
            if label == 0:
                real_correct += 1
            else:
                fake_correct += 1
        
        if label == 0:
            real_total += 1
        else:
            fake_total += 1
    
    accuracy = correct / len(test_samples) * 100
    real_accuracy = real_correct / real_total * 100 if real_total > 0 else 0
    fake_accuracy = fake_correct / fake_total * 100 if fake_total > 0 else 0
    
    print(f"\\nTest Results:")
    print(f"  Overall Accuracy: {correct}/{len(test_samples)} ({accuracy:.1f}%%)")
    print(f"  Real Audio: {real_correct}/{real_total} ({real_accuracy:.1f}%%)")
    print(f"  Fake Audio: {fake_correct}/{fake_total} ({fake_accuracy:.1f}%%)")
    print("="*80)

if __name__ == '__main__':
    main()
""" % {
        'train_target': balanced_counts['train']['target'],
        'val_target': balanced_counts['val']['target'],
        'test_target': balanced_counts['test']['target']
    }
    
    # Save training script
    script_path = PROJECT_ROOT / 'scripts' / 'train_balanced_new.py'
    with open(script_path, 'w') as f:
        f.write(train_script)
    
    print(f"\n✅ Training script created: {script_path}")
    print("\nTo train the model, run:")
    print(f"  python {script_path}")
    
    return script_path

def main():
    print("="*80)
    print("AUDIO DATASET BALANCING AND TRAINING PREPARATION")
    print("="*80)
    
    data_dir = Path('data/audio')
    
    if not data_dir.exists():
        print(f"\n❌ Data directory not found: {data_dir}")
        return
    
    # Scan dataset
    counts = scan_dataset(data_dir)
    
    # Balance dataset
    balanced_counts = balance_dataset(data_dir, counts)
    
    # Create training script
    script_path = train_balanced_model(data_dir, balanced_counts)
    
    print("\n" + "="*80)
    print("READY TO TRAIN")
    print("="*80)
    print("\nNext step: Run the training script to train balanced model")
    print(f"Command: .\\venv\\Scripts\\python.exe {script_path}")

if __name__ == "__main__":
    main()
