"""
Quick training with small balanced audio dataset
Uses only a few samples for fast training and testing
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.audio.model_specRNet import SpecRNetLite
from src.audio.pipeline import preprocess_audio, extract_features

class AudioDataset(Dataset):
    """Audio dataset for training"""
    def __init__(self, file_list, cfg, device):
        self.file_list = file_list
        self.cfg = cfg
        self.device = device
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_path, label = self.file_list[idx]
        
        try:
            # Load and extract features
            wav = preprocess_audio(file_path, self.cfg, self.device)
            features = extract_features(wav, self.cfg)
            return features.squeeze(0).cpu(), torch.FloatTensor([label])
        except Exception as e:
            # Return zeros on error
            return torch.zeros((3, 64, 100)), torch.FloatTensor([label])

def get_small_balanced_samples(data_dir, split, samples_per_class=50):
    """Get small balanced dataset"""
    real_dir = data_dir / split / 'real'
    fake_dir = data_dir / split / 'fake'
    
    real_files = list(real_dir.glob('*.wav'))
    fake_files = list(fake_dir.glob('*.wav'))
    
    # Sample exactly samples_per_class from each class
    real_samples = random.sample(real_files, min(samples_per_class, len(real_files)))
    fake_samples = random.sample(fake_files, min(samples_per_class, len(fake_files)))
    
    # Create combined list with labels
    samples = []
    for f in real_samples:
        samples.append((str(f), 0))  # 0 = real
    for f in fake_samples:
        samples.append((str(f), 1))  # 1 = fake
    
    random.shuffle(samples)
    return samples

def train_epoch(model, train_loader, criterion, optimizer, device, scaler):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for features, labels in pbar:
        features, labels = features.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Don't use autocast with BCELoss
        outputs = model(features)
        loss = criterion(outputs.squeeze(), labels.squeeze())
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        predictions = (outputs.squeeze() > 0.5).float()
        correct += (predictions == labels.squeeze()).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total:.3f}'})
    
    return total_loss / len(train_loader), correct / total

def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    real_correct = 0
    fake_correct = 0
    real_total = 0
    fake_total = 0
    
    with torch.no_grad():
        for features, labels in tqdm(val_loader, desc="Validating"):
            features, labels = features.to(device), labels.to(device)
            
            outputs = model(features)
            loss = criterion(outputs.squeeze(), labels.squeeze())
            
            total_loss += loss.item()
            predictions = (outputs.squeeze() > 0.5).float()
            
            for i, label in enumerate(labels.squeeze()):
                if predictions[i] == label:
                    correct += 1
                    if label == 0:
                        real_correct += 1
                    else:
                        fake_correct += 1
                
                if label == 0:
                    real_total += 1
                else:
                    fake_total += 1
            
            total += labels.size(0)
    
    real_acc = real_correct / real_total if real_total > 0 else 0
    fake_acc = fake_correct / fake_total if fake_total > 0 else 0
    
    return total_loss / len(val_loader), correct / total, real_acc, fake_acc

def main():
    print("="*80)
    print("QUICK AUDIO TRAINING WITH SMALL BALANCED DATASET")
    print("="*80)
    
    # Load config
    cfg = yaml.safe_load(open('configs/hparams.yaml', 'r'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Set random seed
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Data directory
    data_dir = Path('data/audio')
    
    # Get small balanced samples (50 per class for speed)
    print("\nLoading small balanced dataset...")
    train_samples = get_small_balanced_samples(data_dir, 'train', samples_per_class=100)
    val_samples = get_small_balanced_samples(data_dir, 'val', samples_per_class=30)
    test_samples = get_small_balanced_samples(data_dir, 'test', samples_per_class=30)
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_samples)} samples (100 per class)")
    print(f"  Val: {len(val_samples)} samples (30 per class)")
    print(f"  Test: {len(test_samples)} samples (30 per class)")
    
    # Initialize model
    print("\nInitializing SpecRNetLite model...")
    model = SpecRNetLite()
    model = model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training configuration - REDUCED for quick training
    epochs = 15
    batch_size = 32
    learning_rate = 0.0002
    patience = 5
    
    print(f"\nTraining configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Early stopping patience: {patience}")
    
    # Create datasets
    train_dataset = AudioDataset(train_samples, cfg, device)
    val_dataset = AudioDataset(val_samples, cfg, device)
    test_dataset = AudioDataset(test_samples, cfg, device)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size*2, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size*2, shuffle=False, num_workers=0)
    
    # Loss and optimizer - Model outputs probabilities so use BCELoss (not BCEWithLogitsLoss)
    # Disable autocast for BCE to avoid errors
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler(enabled=False)  # Disable mixed precision for BCE
    
    # Create checkpoint directory
    checkpoint_dir = Path('checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)
    
    best_val_acc = 0
    patience_counter = 0
    
    for epoch in range(epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"{'='*80}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        
        # Validate
        val_loss, val_acc, val_real_acc, val_fake_acc = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        print(f"\n  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"  Val Real Acc: {val_real_acc:.4f}, Val Fake Acc: {val_fake_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, checkpoint_dir / 'audio_quick_balanced.pt')
            print(f"  ✅ Best model saved! (val_acc: {val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n  ⚠️  Early stopping triggered (patience={patience})")
                break
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    
    # Load best model for testing
    print("\n" + "="*80)
    print("TESTING ON TEST SET")
    print("="*80)
    
    checkpoint = torch.load(checkpoint_dir / 'audio_quick_balanced.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_real_acc, test_fake_acc = validate(model, test_loader, criterion, device)
    
    print(f"\nTest Results:")
    print(f"  Overall Accuracy: {test_acc:.4f} ({test_acc*100:.1f}%)")
    print(f"  Real Audio Accuracy: {test_real_acc:.4f} ({test_real_acc*100:.1f}%)")
    print(f"  Fake Audio Accuracy: {test_fake_acc:.4f} ({test_fake_acc*100:.1f}%)")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Model saved: checkpoints/audio_quick_balanced.pt")
    print(f"Test Accuracy: {test_acc*100:.1f}%")
    print(f"  - Real: {test_real_acc*100:.1f}%")
    print(f"  - Fake: {test_fake_acc*100:.1f}%")
    
    if test_real_acc > 0.3:  # Better than the 0% from before
        print("\n✅ Model shows improvement on real audio detection!")
    else:
        print("\n⚠️  Model still struggles with real audio detection")
    
    print("="*80)

if __name__ == '__main__':
    main()
