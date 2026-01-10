"""
Test the trained audio model on sample real and fake audio files
"""

import os
import sys
import torch
import yaml
from pathlib import Path

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__) + "/..")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.audio.pipeline import load_audio_model, infer_audio

def predict_audio(audio_path, model, cfg, device):
    """Predict if an audio file is real or fake"""
    try:
        with torch.no_grad():
            score = infer_audio(model, audio_path, cfg, device)
        
        prediction = "FAKE" if score >= 0.5 else "REAL"
        confidence = score if score >= 0.5 else (1 - score)
        
        return prediction, confidence, score
    except Exception as e:
        print(f"❌ Error processing {audio_path}: {e}")
        return None, None, None

def main():
    print("=" * 80)
    print("TESTING TRAINED AUDIO MODEL")
    print("=" * 80)
    
    # Load config
    cfg = yaml.safe_load(open('configs/hparams.yaml', 'r'))
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Load model
    print("Loading audio model...")
    model = load_audio_model(cfg, device, 'checkpoints')
    print("✅ Model loaded!\n")
    
    # Test on sample audio files
    samples_dir = Path("samples")
    
    if not samples_dir.exists():
        print(f"❌ Samples directory not found: {samples_dir}")
        return
    
    # Get audio files
    audio_files = {
        "real": list(samples_dir.glob("real_*.wav")),
        "fake": list(samples_dir.glob("fake_*.wav"))
    }
    
    results = []
    
    print("=" * 80)
    print("TESTING REAL AUDIO SAMPLES")
    print("=" * 80)
    
    for audio_path in audio_files["real"]:
        prediction, confidence, score = predict_audio(audio_path, model, cfg, device)
        
        if prediction:
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
    
    print("\n" + "=" * 80)
    print("TESTING FAKE/AI AUDIO SAMPLES")
    print("=" * 80)
    
    for audio_path in audio_files["fake"]:
        prediction, confidence, score = predict_audio(audio_path, model, cfg, device)
        
        if prediction:
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
            print("✅ EXCELLENT: Model performs very well on these samples!")
        elif accuracy >= 0.7:
            print("✅ GOOD: Model performs reasonably well.")
        elif accuracy >= 0.5:
            print("⚠️  MODERATE: Model performance is above random guessing.")
        else:
            print("❌ POOR: Model needs improvement or more training.")
        print("=" * 80)
    else:
        print("\n⚠️  No samples were successfully tested.")

if __name__ == "__main__":
    main()
