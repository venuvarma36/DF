"""
Download audio samples and test all pretrained models in workspace
"""

import os
import sys
import random
import shutil
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def download_samples_from_dataset(samples_dir, num_per_class=10):
    """Download samples from test dataset"""
    print("="*80)
    print("DOWNLOADING AUDIO SAMPLES FROM DATASET")
    print("="*80)
    
    data_dir = Path("data/audio")
    
    # Get samples from test set
    test_real_dir = data_dir / "test" / "real"
    test_fake_dir = data_dir / "test" / "fake"
    
    if not test_real_dir.exists() or not test_fake_dir.exists():
        print("\n❌ Test dataset not found")
        return 0, 0
    
    # Get files
    real_files = list(test_real_dir.glob("*.wav"))
    fake_files = list(test_fake_dir.glob("*.wav"))
    
    print(f"\nAvailable: {len(real_files)} real, {len(fake_files)} fake")
    print(f"Downloading: {num_per_class} per class\n")
    
    # Sample randomly
    real_samples = random.sample(real_files, min(num_per_class, len(real_files)))
    fake_samples = random.sample(fake_files, min(num_per_class, len(fake_files)))
    
    # Copy to samples directory
    samples_dir.mkdir(exist_ok=True)
    
    real_copied = 0
    fake_copied = 0
    
    print("Copying REAL samples:")
    for i, src in enumerate(real_samples, 1):
        dst = samples_dir / f"test_real_{i}.wav"
        shutil.copy2(src, dst)
        real_copied += 1
        print(f"  ✓ {dst.name}")
    
    print("\nCopying FAKE samples:")
    for i, src in enumerate(fake_samples, 1):
        dst = samples_dir / f"test_fake_{i}.wav"
        shutil.copy2(src, dst)
        fake_copied += 1
        print(f"  ✓ {dst.name}")
    
    print(f"\n✅ Downloaded {real_copied} real + {fake_copied} fake samples")
    
    return real_copied, fake_copied

def test_all_models(samples_dir):
    """Test all audio models in workspace"""
    print("\n" + "="*80)
    print("TESTING ALL AUDIO MODELS")
    print("="*80)
    
    import torch
    import yaml
    from src.audio.pipeline import load_audio_model, infer_audio
    
    # Load config
    cfg = yaml.safe_load(open('configs/hparams.yaml', 'r'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Find all audio model checkpoints
    checkpoint_dir = Path('checkpoints')
    audio_models = list(checkpoint_dir.glob('audio*.pt'))
    
    if not audio_models:
        print("\n❌ No audio models found in checkpoints/")
        return
    
    print(f"\nFound {len(audio_models)} model(s):")
    for model_path in audio_models:
        size_mb = model_path.stat().st_size / 1024 / 1024
        print(f"  - {model_path.name} ({size_mb:.2f} MB)")
    
    # Get test samples
    real_files = sorted(list(samples_dir.glob("test_real_*.wav")))
    fake_files = sorted(list(samples_dir.glob("test_fake_*.wav")))
    
    if not real_files or not fake_files:
        print("\n❌ No test samples found")
        return
    
    print(f"\nTest samples: {len(real_files)} real, {len(fake_files)} fake")
    
    results = []
    
    # Test each model
    for model_path in audio_models:
        print(f"\n{'='*80}")
        print(f"Testing: {model_path.name}")
        print(f"{'='*80}")
        
        try:
            # Load model
            model = load_audio_model(cfg, device, str(checkpoint_dir), model_file=model_path.name)
            
            real_correct = 0
            fake_correct = 0
            real_scores = []
            fake_scores = []
            
            print("\nREAL audio:")
            for audio_file in real_files:
                try:
                    score = infer_audio(model, str(audio_file), cfg, device)
                    real_scores.append(score)
                    prediction = 'FAKE' if score > 0.5 else 'REAL'
                    correct = prediction == 'REAL'
                    real_correct += correct
                    symbol = '✅' if correct else '❌'
                    print(f"  {symbol} {audio_file.name:20s} Score: {score:.4f} → {prediction}")
                except Exception as e:
                    print(f"  ❌ {audio_file.name:20s} Error: {str(e)[:40]}")
            
            print("\nFAKE audio:")
            for audio_file in fake_files:
                try:
                    score = infer_audio(model, str(audio_file), cfg, device)
                    fake_scores.append(score)
                    prediction = 'FAKE' if score > 0.5 else 'REAL'
                    correct = prediction == 'FAKE'
                    fake_correct += correct
                    symbol = '✅' if correct else '❌'
                    print(f"  {symbol} {audio_file.name:20s} Score: {score:.4f} → {prediction}")
                except Exception as e:
                    print(f"  ❌ {audio_file.name:20s} Error: {str(e)[:40]}")
            
            # Calculate metrics
            total = len(real_files) + len(fake_files)
            total_correct = real_correct + fake_correct
            accuracy = total_correct / total * 100 if total > 0 else 0
            real_acc = real_correct / len(real_files) * 100 if real_files else 0
            fake_acc = fake_correct / len(fake_files) * 100 if fake_files else 0
            
            avg_real_score = sum(real_scores) / len(real_scores) if real_scores else 0
            avg_fake_score = sum(fake_scores) / len(fake_scores) if fake_scores else 0
            
            print(f"\n{'='*80}")
            print(f"RESULTS FOR {model_path.name}:")
            print(f"{'='*80}")
            print(f"Overall: {total_correct}/{total} ({accuracy:.1f}%)")
            print(f"Real: {real_correct}/{len(real_files)} ({real_acc:.1f}%)")
            print(f"Fake: {fake_correct}/{len(fake_files)} ({fake_acc:.1f}%)")
            print(f"\nAverage Scores:")
            print(f"  Real samples: {avg_real_score:.4f} (lower is better)")
            print(f"  Fake samples: {avg_fake_score:.4f} (higher is better)")
            print(f"{'='*80}")
            
            results.append({
                'model': model_path.name,
                'accuracy': accuracy,
                'real_acc': real_acc,
                'fake_acc': fake_acc,
                'avg_real_score': avg_real_score,
                'avg_fake_score': avg_fake_score
            })
            
        except Exception as e:
            print(f"\n❌ Failed to test {model_path.name}: {e}")
    
    # Final comparison
    if len(results) > 1:
        print("\n" + "="*80)
        print("MODEL COMPARISON")
        print("="*80)
        
        # Sort by overall accuracy
        results.sort(key=lambda x: x['accuracy'], reverse=True)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['model']}")
            print(f"   Overall: {result['accuracy']:.1f}%")
            print(f"   Real: {result['real_acc']:.1f}% (avg score: {result['avg_real_score']:.4f})")
            print(f"   Fake: {result['fake_acc']:.1f}% (avg score: {result['avg_fake_score']:.4f})")
        
        best = results[0]
        print("\n" + "="*80)
        print(f"✅ BEST MODEL: {best['model']}")
        print(f"   Accuracy: {best['accuracy']:.1f}%")
        print("="*80)

def main():
    print("="*80)
    print("DOWNLOAD SAMPLES & TEST ALL AUDIO MODELS")
    print("="*80)
    
    samples_dir = Path("samples")
    
    # Download samples
    real_count, fake_count = download_samples_from_dataset(samples_dir, num_per_class=10)
    
    if real_count == 0 and fake_count == 0:
        print("\n❌ No samples downloaded")
        return
    
    # Test all models
    test_all_models(samples_dir)

if __name__ == "__main__":
    main()
