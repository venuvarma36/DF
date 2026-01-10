"""
Download more audio samples and test with newly trained audio model
"""

import os
import sys
import shutil
import torch
import yaml
from pathlib import Path
from urllib.request import urlopen
import time

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__) + "/..")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.audio.pipeline import load_audio_model, infer_audio

def download_audio(url, save_path, timeout=30):
    """Download audio file from URL"""
    try:
        with urlopen(url, timeout=timeout) as response:
            with open(save_path, 'wb') as out_file:
                out_file.write(response.read())
        print(f"✅ Downloaded: {os.path.basename(save_path)}")
        return True
    except Exception as e:
        print(f"❌ Failed to download {url}: {e}")
        return False

def get_more_samples():
    """Get more audio samples from dataset"""
    samples_dir = Path("samples")
    samples_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("DOWNLOADING ADDITIONAL AUDIO SAMPLES")
    print("=" * 80)
    print(f"Target directory: {samples_dir.absolute()}\n")
    
    # Copy more samples from test dataset
    test_real_dir = Path("data/audio/test/real")
    test_fake_dir = Path("data/audio/test/fake")
    
    print("📋 Copying more samples from test dataset...\n")
    
    if test_real_dir.exists():
        all_real = list(test_real_dir.glob("*.wav"))
        # Get samples 6-15 (avoiding 1-5 which we already have)
        real_audios = all_real[5:15]
        for i, audio_path in enumerate(real_audios):
            dest = samples_dir / f"real_audio_{i+6}.wav"
            if not dest.exists():
                shutil.copy(audio_path, dest)
                print(f"✅ Copied: {dest.name}")
    
    if test_fake_dir.exists():
        all_fake = list(test_fake_dir.glob("*.wav"))
        # Get samples 6-15 (avoiding 1-5 which we already have)
        fake_audios = all_fake[5:15]
        for i, audio_path in enumerate(fake_audios):
            dest = samples_dir / f"fake_audio_{i+6}.wav"
            if not dest.exists():
                shutil.copy(audio_path, dest)
                print(f"✅ Copied: {dest.name}")
    
    # Summary
    all_samples = list(samples_dir.glob("*.wav"))
    real_count = len([s for s in all_samples if 'real' in s.name.lower()])
    fake_count = len([s for s in all_samples if 'fake' in s.name.lower()])
    
    print(f"\n📊 Total audio samples now: {len(all_samples)}")
    print(f"   - Real audio: {real_count}")
    print(f"   - Fake audio: {fake_count}\n")
    
    return real_count, fake_count

def test_audio_samples():
    """Test with newly trained audio model"""
    print("=" * 80)
    print("TESTING WITH NEWLY TRAINED AUDIO MODEL")
    print("=" * 80)
    
    # Load config
    cfg = yaml.safe_load(open('configs/hparams.yaml', 'r'))
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Load model
    print("Loading trained audio model...")
    model = load_audio_model(cfg, device, 'checkpoints')
    print("✅ Model loaded!\n")
    
    # Test on sample audio files
    samples_dir = Path("samples")
    
    if not samples_dir.exists():
        print(f"❌ Samples directory not found: {samples_dir}")
        return
    
    # Get audio files
    audio_files = {
        "real": sorted(list(samples_dir.glob("real_*.wav"))),
        "fake": sorted(list(samples_dir.glob("fake_*.wav")))
    }
    
    results = []
    
    print("=" * 80)
    print("TESTING REAL AUDIO SAMPLES")
    print("=" * 80)
    
    for audio_path in audio_files["real"]:
        try:
            with torch.no_grad():
                score = infer_audio(model, str(audio_path), cfg, device)
            
            prediction = "FAKE" if score >= 0.5 else "REAL"
            confidence = score if score >= 0.5 else (1 - score)
            
            expected = "REAL"
            correct = "✅" if prediction == expected else "❌"
            
            results.append({
                'file': audio_path.name,
                'expected': expected,
                'predicted': prediction,
                'confidence': confidence,
                'score': score,
                'correct': correct
            })
            
            print(f"{correct} {audio_path.name:25s} | "
                  f"Expected: {expected:4s} | "
                  f"Predicted: {prediction:4s} ({confidence:.2%}) | "
                  f"Score: {score:.4f}")
        except Exception as e:
            print(f"❌ Error testing {audio_path.name}: {e}")
    
    print("\n" + "=" * 80)
    print("TESTING FAKE/AI AUDIO SAMPLES")
    print("=" * 80)
    
    for audio_path in audio_files["fake"]:
        try:
            with torch.no_grad():
                score = infer_audio(model, str(audio_path), cfg, device)
            
            prediction = "FAKE" if score >= 0.5 else "REAL"
            confidence = score if score >= 0.5 else (1 - score)
            
            expected = "FAKE"
            correct = "✅" if prediction == expected else "❌"
            
            results.append({
                'file': audio_path.name,
                'expected': expected,
                'predicted': prediction,
                'confidence': confidence,
                'score': score,
                'correct': correct
            })
            
            print(f"{correct} {audio_path.name:25s} | "
                  f"Expected: {expected:4s} | "
                  f"Predicted: {prediction:4s} ({confidence:.2%}) | "
                  f"Score: {score:.4f}")
        except Exception as e:
            print(f"❌ Error testing {audio_path.name}: {e}")
    
    # Calculate metrics
    if results:
        print("\n" + "=" * 80)
        print("PERFORMANCE METRICS")
        print("=" * 80)
        
        total = len(results)
        correct = sum(1 for r in results if r['correct'] == '✅')
        accuracy = correct / total if total > 0 else 0
        
        # Calculate by class
        real_results = [r for r in results if r['expected'] == 'REAL']
        fake_results = [r for r in results if r['expected'] == 'FAKE']
        
        real_correct = sum(1 for r in real_results if r['correct'] == '✅')
        fake_correct = sum(1 for r in fake_results if r['correct'] == '✅')
        
        real_accuracy = real_correct / len(real_results) if real_results else 0
        fake_accuracy = fake_correct / len(fake_results) if fake_results else 0
        
        print(f"Total samples tested: {total}")
        print(f"Correct predictions: {correct}")
        print(f"Overall Accuracy: {accuracy:.2%}\n")
        
        print(f"Real Audio (Human Speech):")
        print(f"  - Samples: {len(real_results)}")
        print(f"  - Correct: {real_correct}")
        print(f"  - Accuracy: {real_accuracy:.2%}\n")
        
        print(f"Fake Audio (AI Generated):")
        print(f"  - Samples: {len(fake_results)}")
        print(f"  - Correct: {fake_correct}")
        print(f"  - Accuracy: {fake_accuracy:.2%}\n")
        
        # Average scores
        real_avg_score = sum(r['score'] for r in real_results) / len(real_results) if real_results else 0
        fake_avg_score = sum(r['score'] for r in fake_results) / len(fake_results) if fake_results else 0
        
        print(f"Average Fake Score:")
        print(f"  - Real audio: {real_avg_score:.4f} (lower is better)")
        print(f"  - Fake audio: {fake_avg_score:.4f} (higher is better)")
        
        print("\n" + "=" * 80)
        if accuracy >= 0.9:
            print("✅ EXCELLENT: Model performs very well!")
        elif accuracy >= 0.8:
            print("✅ VERY GOOD: Model is well-trained!")
        elif accuracy >= 0.7:
            print("✅ GOOD: Model performs reasonably well.")
        elif accuracy >= 0.6:
            print("✅ FAIR: Model performance is improving.")
        else:
            print("⚠️  NEEDS IMPROVEMENT: Consider more training or data.")
        print("=" * 80)
    else:
        print("\n⚠️  No samples were successfully tested.")

def main():
    print("\n")
    get_more_samples()
    
    print("⏳ Waiting for training to complete if still running...")
    time.sleep(5)
    
    test_audio_samples()

if __name__ == "__main__":
    main()
