"""
Monitor the training progress of the Kaggle audio deepfake model
"""
import os
import json
from pathlib import Path
from datetime import datetime
import time

LOG_PATH = Path(r'C:\DeepFake_detection\logs')
CHECKPOINT_PATH = Path(r'C:\DeepFake_detection\checkpoints')

print("\n" + "="*80)
print("📊 KAGGLE AUDIO DEEPFAKE MODEL - TRAINING MONITOR")
print("="*80)
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Log directory: {LOG_PATH}")
print(f"Checkpoint directory: {CHECKPOINT_PATH}")

# Check for model files
model_files = list(CHECKPOINT_PATH.glob('audio_kaggle*.pt'))
if model_files:
    print(f"\n📁 Model Checkpoints:")
    for mf in sorted(model_files):
        size_mb = mf.stat().st_size / (1024*1024)
        print(f"  ✓ {mf.name} ({size_mb:.2f} MB)")

# Check for training history
history_files = list(LOG_PATH.glob('training_history_*.json'))
if history_files:
    latest_history = sorted(history_files)[-1]
    print(f"\n📈 Latest Training History: {latest_history.name}")
    
    with open(latest_history, 'r') as f:
        history = json.load(f)
    
    epochs = len(history['train_loss'])
    print(f"  Epochs completed: {epochs}")
    print(f"  Latest metrics:")
    print(f"    Train Loss: {history['train_loss'][-1]:.4f}")
    print(f"    Train Acc:  {history['train_acc'][-1]:.4f}")
    print(f"    Val Loss:   {history['val_loss'][-1]:.4f}")
    print(f"    Val Acc:    {history['val_acc'][-1]:.4f}")
    print(f"  Best Val Loss: {min(history['val_loss']):.4f} at epoch {history['val_loss'].index(min(history['val_loss'])) + 1}")

# Check for test results
results_files = list(LOG_PATH.glob('test_results_*.json'))
if results_files:
    latest_results = sorted(results_files)[-1]
    print(f"\n✅ Latest Test Results: {latest_results.name}")
    
    with open(latest_results, 'r') as f:
        results = json.load(f)
    
    print(f"  Dataset: {results['dataset']}")
    print(f"  Test Accuracy:     {results['test_accuracy']*100:.2f}%")
    print(f"  Real Accuracy:     {results['real_accuracy']*100:.2f}%")
    print(f"  Fake Accuracy:     {results['fake_accuracy']*100:.2f}%")
    print(f"  Training Epochs:   {results['training_epochs']}")
    print(f"  Class Weights:")
    print(f"    Real: {results['class_weights']['real']:.4f}")
    print(f"    Fake: {results['class_weights']['fake']:.4f}")

print("\n" + "="*80 + "\n")
