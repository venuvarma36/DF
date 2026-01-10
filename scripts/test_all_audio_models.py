"""
Test all available audio deepfake detection models on downloaded samples
Includes custom trained model and any available pretrained models
"""

import os
import sys
import torch
import yaml
import librosa
import numpy as np
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.audio.pipeline import load_audio_model, infer_audio

def test_custom_model(samples_dir):
    """Test the custom trained audio model"""
    print("\n" + "="*80)
    print("TESTING CUSTOM TRAINED MODEL (SpecRNet-Lite)")
    print("="*80)
    
    try:
        # Load config and model
        cfg = yaml.safe_load(open('configs/hparams.yaml', 'r'))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading model on {device}...")
        model = load_audio_model(cfg, device, 'checkpoints')
        print("✅ Model loaded successfully!\n")
        
        # Get audio files
        real_files = sorted(list(samples_dir.glob('real_audio_*.wav')))
        fake_files = sorted(list(samples_dir.glob('fake_audio_*.wav')))
        
        print(f"Testing on {len(real_files)} REAL and {len(fake_files)} FAKE samples\n")
        
        # Test real audio
        real_correct = 0
        real_scores = []
        print("Testing REAL audio samples:")
        for audio_file in real_files:
            try:
                score = infer_audio(model, str(audio_file), cfg, device)
                real_scores.append(score)
                prediction = 'FAKE' if score > 0.5 else 'REAL'
                correct = prediction == 'REAL'
                real_correct += correct
                symbol = '✅' if correct else '❌'
                print(f'{symbol} {audio_file.name:30s} Score: {score:.4f} → {prediction}')
            except Exception as e:
                print(f'❌ {audio_file.name:30s} Error: {str(e)[:40]}')
        
        # Test fake audio
        fake_correct = 0
        fake_scores = []
        print("\nTesting FAKE audio samples:")
        for audio_file in fake_files:
            try:
                score = infer_audio(model, str(audio_file), cfg, device)
                fake_scores.append(score)
                prediction = 'FAKE' if score > 0.5 else 'REAL'
                correct = prediction == 'FAKE'
                fake_correct += correct
                symbol = '✅' if correct else '❌'
                print(f'{symbol} {audio_file.name:30s} Score: {score:.4f} → {prediction}')
            except Exception as e:
                print(f'❌ {audio_file.name:30s} Error: {str(e)[:40]}')
        
        # Calculate metrics
        total_samples = len(real_files) + len(fake_files)
        total_correct = real_correct + fake_correct
        overall_accuracy = (total_correct / total_samples * 100) if total_samples > 0 else 0
        real_accuracy = (real_correct / len(real_files) * 100) if real_files else 0
        fake_accuracy = (fake_correct / len(fake_files) * 100) if fake_files else 0
        
        avg_real_score = np.mean(real_scores) if real_scores else 0
        avg_fake_score = np.mean(fake_scores) if fake_scores else 0
        
        print(f'\n{"="*80}')
        print("CUSTOM MODEL RESULTS:")
        print(f'{"="*80}')
        print(f'Real Audio Accuracy: {real_correct}/{len(real_files)} ({real_accuracy:.1f}%)')
        print(f'Fake Audio Accuracy: {fake_correct}/{len(fake_files)} ({fake_accuracy:.1f}%)')
        print(f'Overall Accuracy: {total_correct}/{total_samples} ({overall_accuracy:.1f}%)')
        print(f'\nAverage Scores:')
        print(f'  Real samples: {avg_real_score:.4f} (lower is better for REAL)')
        print(f'  Fake samples: {avg_fake_score:.4f} (higher is better for FAKE)')
        print(f'{"="*80}')
        
        return {
            'model': 'Custom SpecRNet-Lite',
            'accuracy': overall_accuracy,
            'real_accuracy': real_accuracy,
            'fake_accuracy': fake_accuracy,
            'total_correct': total_correct,
            'total_samples': total_samples
        }
        
    except Exception as e:
        print(f"❌ Failed to test custom model: {e}")
        return None

def test_huggingface_model(model_name, samples_dir):
    """Test a HuggingFace pretrained model"""
    print(f"\n{'='*80}")
    print(f"TESTING HUGGINGFACE MODEL: {model_name}")
    print(f"{'='*80}")
    
    try:
        from transformers import pipeline
        
        print("Loading model...")
        device = 0 if torch.cuda.is_available() else -1
        classifier = pipeline("audio-classification", model=model_name, device=device)
        print("✅ Model loaded successfully!\n")
        
        # Get audio files
        real_files = sorted(list(samples_dir.glob('real_audio_*.wav')))
        fake_files = sorted(list(samples_dir.glob('fake_audio_*.wav')))
        
        print(f"Testing on {len(real_files)} REAL and {len(fake_files)} FAKE samples\n")
        
        results = []
        
        # Test real audio
        print("Testing REAL audio samples:")
        for audio_file in real_files:
            try:
                audio, sr = librosa.load(str(audio_file), sr=16000)
                result = classifier(audio, sampling_rate=sr)
                
                # Get prediction
                label = result[0]['label'].upper()
                confidence = result[0]['score']
                
                expected = "REAL"
                correct = label == expected
                symbol = '✅' if correct else '❌'
                
                results.append({
                    'expected': expected,
                    'predicted': label,
                    'correct': correct
                })
                
                print(f'{symbol} {audio_file.name:30s} Pred: {label:6s} ({confidence:.2%})')
            except Exception as e:
                print(f'❌ {audio_file.name:30s} Error: {str(e)[:40]}')
        
        # Test fake audio
        print("\nTesting FAKE audio samples:")
        for audio_file in fake_files:
            try:
                audio, sr = librosa.load(str(audio_file), sr=16000)
                result = classifier(audio, sampling_rate=sr)
                
                # Get prediction
                label = result[0]['label'].upper()
                confidence = result[0]['score']
                
                expected = "FAKE"
                correct = label == expected
                symbol = '✅' if correct else '❌'
                
                results.append({
                    'expected': expected,
                    'predicted': label,
                    'correct': correct
                })
                
                print(f'{symbol} {audio_file.name:30s} Pred: {label:6s} ({confidence:.2%})')
            except Exception as e:
                print(f'❌ {audio_file.name:30s} Error: {str(e)[:40]}')
        
        # Calculate metrics
        total_samples = len(results)
        total_correct = sum(1 for r in results if r['correct'])
        
        real_results = [r for r in results if r['expected'] == 'REAL']
        fake_results = [r for r in results if r['expected'] == 'FAKE']
        
        real_correct = sum(1 for r in real_results if r['correct'])
        fake_correct = sum(1 for r in fake_results if r['correct'])
        
        overall_accuracy = (total_correct / total_samples * 100) if total_samples > 0 else 0
        real_accuracy = (real_correct / len(real_results) * 100) if real_results else 0
        fake_accuracy = (fake_correct / len(fake_results) * 100) if fake_results else 0
        
        print(f'\n{"="*80}')
        print(f"RESULTS FOR {model_name}:")
        print(f'{"="*80}')
        print(f'Real Audio Accuracy: {real_correct}/{len(real_results)} ({real_accuracy:.1f}%)')
        print(f'Fake Audio Accuracy: {fake_correct}/{len(fake_results)} ({fake_accuracy:.1f}%)')
        print(f'Overall Accuracy: {total_correct}/{total_samples} ({overall_accuracy:.1f}%)')
        print(f'{"="*80}')
        
        return {
            'model': model_name,
            'accuracy': overall_accuracy,
            'real_accuracy': real_accuracy,
            'fake_accuracy': fake_accuracy,
            'total_correct': total_correct,
            'total_samples': total_samples
        }
        
    except Exception as e:
        print(f"❌ Failed to test {model_name}: {e}")
        return None

def main():
    print("="*80)
    print("COMPREHENSIVE AUDIO DEEPFAKE DETECTION MODEL TESTING")
    print("="*80)
    
    samples_dir = Path("samples")
    if not samples_dir.exists():
        print(f"❌ Samples directory not found: {samples_dir}")
        return
    
    # Count samples
    real_count = len(list(samples_dir.glob('real_audio_*.wav')))
    fake_count = len(list(samples_dir.glob('fake_audio_*.wav')))
    print(f"\nTotal samples: {real_count} REAL, {fake_count} FAKE\n")
    
    results_list = []
    
    # Test custom model
    custom_result = test_custom_model(samples_dir)
    if custom_result:
        results_list.append(custom_result)
    
    # Try testing HuggingFace models (optional)
    print("\n" + "="*80)
    print("Would you like to test HuggingFace pretrained models? (Takes time)")
    print("Skipping HuggingFace models for now (enable by editing script)")
    print("="*80)
    
    # Uncomment to test HuggingFace models:
    # hf_models = [
    #     "microsoft/wavlm-base-plus",
    #     "facebook/wav2vec2-base-960h",
    # ]
    # for model_name in hf_models:
    #     result = test_huggingface_model(model_name, samples_dir)
    #     if result:
    #         results_list.append(result)
    
    # Final summary
    if results_list:
        print("\n" + "="*80)
        print("FINAL SUMMARY - ALL MODELS")
        print("="*80)
        
        results_list.sort(key=lambda x: x['accuracy'], reverse=True)
        
        for i, result in enumerate(results_list, 1):
            print(f"\n{i}. {result['model']}")
            print(f"   Overall: {result['accuracy']:.1f}% ({result['total_correct']}/{result['total_samples']})")
            print(f"   Real: {result['real_accuracy']:.1f}%")
            print(f"   Fake: {result['fake_accuracy']:.1f}%")
        
        if results_list:
            best = results_list[0]
            print("\n" + "="*80)
            print(f"✅ BEST PERFORMING MODEL: {best['model']}")
            print(f"   Overall Accuracy: {best['accuracy']:.1f}%")
            print("="*80)

if __name__ == "__main__":
    main()
