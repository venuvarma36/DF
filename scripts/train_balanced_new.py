
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
from src.audio.feature_extractor import extract_mel_spectrogram

class AudioDataset(Dataset):
    """Audio dataset for training"""
    def __init__(self, file_list, cfg):
        self.file_list = file_list
        self.cfg = cfg
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_path, label = self.file_list[idx]
        
        try:
            # Extract features
            features = extract_mel_spectrogram(file_path, self.cfg)
            return torch.FloatTensor(features), torch.FloatTensor([label])
        except Exception as e:
            # Return zeros on error
            return torch.zeros((3, 64, 100)), torch.FloatTensor([label])

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
        
        with torch.amp.autocast(device_type='cuda', enabled=True):
            outputs = model(features)
            loss = criterion(outputs.squeeze(), labels.squeeze())
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        predictions = (outputs.squeeze() > 0.5).float()
        correct += (predictions == labels.squeeze()).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({'loss': loss.item(), 'acc': correct/total})
    
    return total_loss / len(train_loader), correct / total

def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for features, labels in tqdm(val_loader, desc="Validating"):
            features, labels = features.to(device), labels.to(device)
            
            outputs = model(features)
            loss = criterion(outputs.squeeze(), labels.squeeze())
            
            total_loss += loss.item()
            predictions = (outputs.squeeze() > 0.5).float()
            correct += (predictions == labels.squeeze()).sum().item()
            total += labels.size(0)
    
    return total_loss / len(val_loader), correct / total

def get_balanced_samples(data_dir, split, target_count):
    """Get balanced samples from dataset"""
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
    print(f"\nDevice: {device}")
    
    # Set random seed
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Data directory
    data_dir = Path('data/audio')
    
    # Balanced counts from scan
    balanced_counts = {
        'train': {'target': 644},
        'val': {'target': 150},
        'test': {'target': 143}
    }
    
    print(f"\nBalanced dataset:")
    print(f"  Train: {balanced_counts['train']['target']*2} samples")
    print(f"  Val: {balanced_counts['val']['target']*2} samples")
    print(f"  Test: {balanced_counts['test']['target']*2} samples")
    
    # Get balanced samples
    train_samples = get_balanced_samples(data_dir, 'train', balanced_counts['train']['target'])
    val_samples = get_balanced_samples(data_dir, 'val', balanced_counts['val']['target'])
    test_samples = get_balanced_samples(data_dir, 'test', balanced_counts['test']['target'])
    
    print(f"\nActual loaded samples:")
    print(f"  Train: {len(train_samples)}")
    print(f"  Val: {len(val_samples)}")
    print(f"  Test: {len(test_samples)}")
    
    # Initialize model
    print("\nInitializing SpecRNetLite model...")
    model = SpecRNetLite()
    model = model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training configuration
    epochs = 25
    batch_size = 64
    learning_rate = 0.0001
    patience = 5
    
    print(f"\nTraining configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Early stopping patience: {patience}")
    print(f"  Mixed precision: FP16")
    
    # Create datasets
    train_dataset = AudioDataset(train_samples, cfg)
    val_dataset = AudioDataset(val_samples, cfg)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size*2, shuffle=False, num_workers=0)
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler(enabled=True)
    
    # Create checkpoint directory
    checkpoint_dir = Path('checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)
    
    best_val_acc = 0
    patience_counter = 0
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, checkpoint_dir / 'audio_balanced_new.pt')
            print(f"  ✅ Best model saved (val_acc: {val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n  Early stopping triggered (patience={patience})")
                break
    
    # Load best model
    checkpoint = torch.load(checkpoint_dir / 'audio_balanced_new.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"\nBest model saved to: checkpoints/audio_balanced_new.pt")
    
    # Test on test set
    print("\n" + "="*80)
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
    
    print(f"\nTest Results:")
    print(f"  Overall Accuracy: {correct}/{len(test_samples)} ({accuracy:.1f}%)")
    print(f"  Real Audio: {real_correct}/{real_total} ({real_accuracy:.1f}%)")
    print(f"  Fake Audio: {fake_correct}/{fake_total} ({fake_accuracy:.1f}%)")
    print("="*80)

if __name__ == '__main__':
    main()
