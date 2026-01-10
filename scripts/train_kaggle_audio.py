"""
Train audio deepfake detector on Kaggle In-The-Wild dataset with class weighting
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchaudio
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime
import sys

# Add src to path
sys.path.insert(0, r'C:\DeepFake_detection')

from src.audio.model_specRNet import SpecRNetLite
from src.audio.pipeline import MultiResSpectrogram, extract_features

# Configuration
CONFIG = {
    'sample_rate': 16000,
    'n_mels': 80,
    'n_fft': 400,
    'hop_length': 160,
    'duration_sec': 5.0,
    'batch_size': 32,
    'epochs': 20,
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'early_stopping_patience': 5,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

# Data paths
DATA_PATH = Path(r'C:\DeepFake_detection\data\audio')
CHECKPOINT_PATH = Path(r'C:\DeepFake_detection\checkpoints')
LOG_PATH = Path(r'C:\DeepFake_detection\logs')

CHECKPOINT_PATH.mkdir(exist_ok=True)
LOG_PATH.mkdir(exist_ok=True)

# Class to handle audio loading and preprocessing
class AudioDataset(Dataset):
    def __init__(self, split='train', transform=None):
        self.split = split
        self.transform = transform
        self.audio_files = []
        self.labels = []
        
        # Load file paths and labels
        for label, label_name in [(0, 'real'), (1, 'fake')]:
            label_dir = DATA_PATH / split / label_name
            if label_dir.exists():
                files = list(label_dir.glob('*.wav'))
                self.audio_files.extend(files)
                self.labels.extend([label] * len(files))
        
        print(f"  {split.upper()}: {len(self.audio_files)} files ({self.labels.count(0)} real, {self.labels.count(1)} fake)")
        
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        label = self.labels[idx]
        
        try:
            # Load audio
            waveform, sr = torchaudio.load(str(audio_path))
            
            # Resample if necessary
            if sr != CONFIG['sample_rate']:
                resampler = torchaudio.transforms.Resample(sr, CONFIG['sample_rate'])
                waveform = resampler(waveform)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # Normalize
            if waveform.abs().max() > 0:
                waveform = waveform / waveform.abs().max()
            
            # Fixed 5s crop/pad
            target_len = int(CONFIG['duration_sec'] * CONFIG['sample_rate'])
            cur_len = waveform.shape[1]
            if cur_len > target_len:
                waveform = waveform[:, :target_len]
            elif cur_len < target_len:
                pad = torch.zeros(1, target_len - cur_len)
                waveform = torch.cat([waveform, pad], dim=1)
            
            return waveform, label
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            # Return silence on error
            return torch.zeros(1, int(CONFIG['duration_sec'] * CONFIG['sample_rate'])), label

def collate_fn(batch):
    """Custom collate function to handle variable length audio"""
    waveforms, labels = zip(*batch)
    
    # Pad to max length in batch
    max_len = max(w.shape[1] for w in waveforms)
    padded = []
    for w in waveforms:
        if w.shape[1] < max_len:
            pad = torch.zeros(1, max_len - w.shape[1])
            w = torch.cat([w, pad], dim=1)
        padded.append(w[:, :max_len])
    
    waveforms = torch.cat(padded, dim=0)
    labels = torch.tensor(labels, dtype=torch.float32)
    
    return waveforms, labels

class AudioTrainer:
    def __init__(self):
        self.device = torch.device(CONFIG['device'])
        print(f"\n🔧 Device: {self.device}")
        
        # Load datasets
        print("\n📊 Loading datasets...")
        self.train_dataset = AudioDataset('train')
        self.val_dataset = AudioDataset('val')
        self.test_dataset = AudioDataset('test')
        
        # Calculate class weights (for reporting) and build sampler for balanced training
        train_labels = np.array(self.train_dataset.labels)
        class_counts = np.bincount(train_labels)
        class_weights = len(train_labels) / (len(np.unique(train_labels)) * class_counts)
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.device)

        # Weighted sampler (inverse frequency)
        inv_freq = {c: 1.0 / class_counts[c] for c in range(len(class_counts))}
        sample_weights = torch.tensor([inv_freq[l] for l in train_labels], dtype=torch.double)
        from torch.utils.data import WeightedRandomSampler
        self.train_sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

        print(f"\n⚖️  Class Weights (reporting):")
        print(f"   Real (0): {self.class_weights[0]:.4f}")
        print(f"   Fake (1): {self.class_weights[1]:.4f}")
        print("⚖️  Using WeightedRandomSampler for balanced training")
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=CONFIG['batch_size'], 
            sampler=self.train_sampler,
            collate_fn=collate_fn,
            num_workers=0
        )
        self.val_loader = DataLoader(
            self.val_dataset, 
            batch_size=CONFIG['batch_size'], 
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )
        self.test_loader = DataLoader(
            self.test_dataset, 
            batch_size=CONFIG['batch_size'], 
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )
        
        # Initialize model
        print("\n🧠 Initializing SpecRNetLite...")
        self.model = SpecRNetLite(in_branches=1).to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"   Total parameters: {total_params:,}")
        
        # Loss and optimizer
        self.criterion = nn.BCELoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=CONFIG['learning_rate'],
            weight_decay=CONFIG['weight_decay']
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=2, 
            verbose=True
        )
        
        # Training state
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
        }
    
    def extract_features(self, waveform):
        """Extract single-branch mel-spectrogram [B, 1, F, T] with requested params"""
        mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=CONFIG['sample_rate'],
            n_mels=CONFIG['n_mels'],
            n_fft=CONFIG['n_fft'],
            hop_length=CONFIG['hop_length'],
            power=2.0,
            center=True,
            pad_mode='reflect',
        ).to(self.device)(waveform)
        mel_spec = torch.clamp(mel_spec, min=1e-9)
        mel_spec = torch.log(mel_spec)
        if mel_spec.dim() == 3:
            mel_spec = mel_spec.unsqueeze(1)
        return mel_spec
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for waveforms, labels in pbar:
            waveforms = waveforms.to(self.device)
            labels = labels.to(self.device).unsqueeze(1)
            
            # Extract features
            spectrograms = self.extract_features(waveforms)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(spectrograms)
            
            # Loss (outputs are already probabilities from sigmoid)
            loss = self.criterion(outputs, labels.squeeze())
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            preds = (outputs > 0.5).float()
            correct += (preds == labels.squeeze()).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        epoch_loss = total_loss / len(self.train_loader)
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validating')
            for waveforms, labels in pbar:
                waveforms = waveforms.to(self.device)
                labels = labels.to(self.device).unsqueeze(1)
                
                # Extract features
                spectrograms = self.extract_features(waveforms)
                
                # Forward pass
                outputs = self.model(spectrograms)
                loss = self.criterion(outputs, labels.squeeze())
                
                # Metrics
                total_loss += loss.item()
                preds = (outputs > 0.5).float()
                correct += (preds == labels.squeeze()).sum().item()
                total += labels.size(0)
                
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        epoch_loss = total_loss / len(self.val_loader)
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def test(self):
        self.model.eval()
        correct = 0
        total = 0
        real_correct = 0
        real_total = 0
        fake_correct = 0
        fake_total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc='Testing')
            for waveforms, labels in pbar:
                waveforms = waveforms.to(self.device)
                labels = labels.to(self.device).unsqueeze(1)
                
                # Extract features
                spectrograms = self.extract_features(waveforms)
                
                # Forward pass
                outputs = self.model(spectrograms)
                preds = (outputs > 0.5).float()
                
                # Overall accuracy
                correct += (preds == labels.squeeze()).sum().item()
                total += labels.size(0)
                
                # Per-class accuracy
                for i in range(labels.size(0)):
                    if labels[i] == 0:  # Real
                        real_total += 1
                        if preds[i] == 0:
                            real_correct += 1
                    else:  # Fake
                        fake_total += 1
                        if preds[i] == 1:
                            fake_correct += 1
                
                pbar.set_postfix({'acc': f"{correct / total:.4f}"})
        
        test_acc = correct / total
        real_acc = real_correct / real_total if real_total > 0 else 0
        fake_acc = fake_correct / fake_total if fake_total > 0 else 0
        
        return test_acc, real_acc, fake_acc
    
    def train(self):
        print(f"\n{'='*80}")
        print(f"TRAINING AUDIO DEEPFAKE DETECTOR")
        print(f"{'='*80}")
        print(f"Dataset: Kaggle In-The-Wild Audio Deepfake")
        print(f"Training samples: {len(self.train_dataset)}")
        print(f"Validation samples: {len(self.val_dataset)}")
        print(f"Test samples: {len(self.test_dataset)}")
        print(f"Batch size: {CONFIG['batch_size']}")
        print(f"Epochs: {CONFIG['epochs']}")
        print(f"Device: {CONFIG['device']}")
        print(f"{'='*80}\n")
        
        for epoch in range(CONFIG['epochs']):
            print(f"\nEpoch {epoch+1}/{CONFIG['epochs']}")
            print("-" * 80)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Early stopping and checkpoint
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                
                # Save best model
                checkpoint_path = CHECKPOINT_PATH / 'audio_kaggle_best.pt'
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f"✓ Best model saved: {checkpoint_path}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= CONFIG['early_stopping_patience']:
                    print(f"\n⏹️  Early stopping triggered (patience={CONFIG['early_stopping_patience']})")
                    break
        
        # Save final model
        final_path = CHECKPOINT_PATH / 'audio_kaggle_final.pt'
        torch.save(self.model.state_dict(), final_path)
        print(f"\n✓ Final model saved: {final_path}")
        
        # Save training history
        history_path = LOG_PATH / f'training_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"✓ Training history saved: {history_path}")
        
        # Test
        print(f"\n{'='*80}")
        print("TESTING ON TEST SET")
        print(f"{'='*80}\n")
        
        # Load best model for testing
        best_model_path = CHECKPOINT_PATH / 'audio_kaggle_best.pt'
        self.model.load_state_dict(torch.load(best_model_path))
        
        test_acc, real_acc, fake_acc = self.test()
        
        print(f"\n{'='*80}")
        print("TEST RESULTS")
        print(f"{'='*80}")
        print(f"Overall Accuracy:  {test_acc*100:.2f}%")
        print(f"Real Accuracy:     {real_acc*100:.2f}%")
        print(f"Fake Accuracy:     {fake_acc*100:.2f}%")
        print(f"{'='*80}\n")
        
        # Save results
        results = {
            'best_val_loss': float(self.best_val_loss),
            'test_accuracy': float(test_acc),
            'real_accuracy': float(real_acc),
            'fake_accuracy': float(fake_acc),
            'training_epochs': epoch + 1,
            'class_weights': {
                'real': float(self.class_weights[0]),
                'fake': float(self.class_weights[1]),
            },
            'dataset': 'Kaggle In-The-Wild Audio Deepfake',
            'timestamp': datetime.now().isoformat(),
        }
        
        results_path = LOG_PATH / f'test_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✓ Test results saved: {results_path}\n")

if __name__ == '__main__':
    trainer = AudioTrainer()
    trainer.train()
