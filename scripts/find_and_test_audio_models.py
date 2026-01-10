"""
Find and test pretrained deepfake audio detection models
Automatically installs missing dependencies and tests on samples
"""

import os
import sys
import subprocess
from pathlib import Path

# First, check and install required dependencies
def check_and_install_dependencies():
    """Check if all required packages are installed, install if missing"""
    print("Checking dependencies...")
    
    required_packages = {
        'librosa': 'librosa',
        'soundfile': 'soundfile',
        'scipy': 'scipy',
        'numpy': 'numpy',
        'torch': 'torch',
        'transformers': 'transformers',
        'PIL': 'Pillow',
    }
    
    missing = []
    
    for module_name, package_name in required_packages.items():
        try:
            __import__(module_name)
            print(f"  ✓ {module_name}")
        except ImportError:
            print(f"  ✗ {module_name} - MISSING")
            missing.append(package_name)
    
    if missing:
        print(f"\nInstalling missing packages: {', '.join(missing)}")
        for package in missing:
            print(f"  Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])
            print(f"  ✓ {package} installed")
    
    print("✓ All dependencies ready!\n")

# Now import everything
import torch
import librosa
import numpy as np
from pathlib import Path
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

def test_model(model_name, samples_dir, model_type="audio_classification"):
    """Test a specific model on samples"""
    print(f"\n{'='*80}")
    print(f"Testing Model: {model_name}")
    print(f"Type: {model_type}")
    print(f"{'='*80}")
    
    try:
        print("📥 Loading model...")
        
        if model_type == "audio_classification":
            classifier = pipeline("audio-classification", model=model_name, device=0 if torch.cuda.is_available() else -1)
        elif model_type == "text-classification":
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            classifier = pipeline("text-classification", model=model_name)
        else:
            print("❌ Unknown model type")
            return None
        
        print("✅ Model loaded!")
        
        # Get audio files
        real_audio_files = sorted(list(samples_dir.glob("real_audio_*.wav")))[:5]
        fake_audio_files = sorted(list(samples_dir.glob("fake_audio_*.wav")))[:5]
        
        if not real_audio_files or not fake_audio_files:
            print("⚠️  No sample audio files found")
            return None
        
        results = []
        
        print("\nTesting REAL audio samples:")
        for audio_path in real_audio_files:
            try:
                # Load audio
                audio, sr = librosa.load(str(audio_path), sr=16000)
                
                # Run classifier
                result = classifier(audio, sampling_rate=sr)
                prediction = result[0]['label'].upper()
                confidence = result[0]['score']
                
                expected = "REAL"
                correct = "✅" if prediction == expected else "❌"
                
                results.append({
                    'file': audio_path.name,
                    'expected': expected,
                    'predicted': prediction,
                    'confidence': confidence,
                    'correct': correct
                })
                
                print(f"{correct} {audio_path.name:20s} | Pred: {prediction:6s} ({confidence:.2%})")
                
            except Exception as e:
                print(f"❌ Error with {audio_path.name}: {str(e)[:50]}")
        
        print("\nTesting FAKE audio samples:")
        for audio_path in fake_audio_files:
            try:
                # Load audio
                audio, sr = librosa.load(str(audio_path), sr=16000)
                
                # Run classifier
                result = classifier(audio, sampling_rate=sr)
                prediction = result[0]['label'].upper()
                confidence = result[0]['score']
                
                expected = "FAKE"
                correct = "✅" if prediction == expected else "❌"
                
                results.append({
                    'file': audio_path.name,
                    'expected': expected,
                    'predicted': prediction,
                    'confidence': confidence,
                    'correct': correct
                })
                
                print(f"{correct} {audio_path.name:20s} | Pred: {prediction:6s} ({confidence:.2%})")
                
            except Exception as e:
                print(f"❌ Error with {audio_path.name}: {str(e)[:50]}")
        
        # Calculate metrics
        if results:
            total = len(results)
            correct = sum(1 for r in results if r['correct'] == '✅')
            accuracy = correct / total if total > 0 else 0
            
            real_results = [r for r in results if r['expected'] == 'REAL']
            fake_results = [r for r in results if r['expected'] == 'FAKE']
            
            real_correct = sum(1 for r in real_results if r['correct'] == '✅')
            fake_correct = sum(1 for r in fake_results if r['correct'] == '✅')
            
            real_acc = real_correct / len(real_results) if real_results else 0
            fake_acc = fake_correct / len(fake_results) if fake_results else 0
            
            print(f"\n{'='*80}")
            print("RESULTS:")
            print(f"{'='*80}")
            print(f"Overall Accuracy: {accuracy:.2%} ({correct}/{total})")
            print(f"Real Audio Accuracy: {real_acc:.2%}")
            print(f"Fake Audio Accuracy: {fake_acc:.2%}")
            print(f"{'='*80}")
            
            return {
                'model': model_name,
                'accuracy': accuracy,
                'real_accuracy': real_acc,
                'fake_accuracy': fake_acc,
                'total_correct': correct,
                'total_samples': total
            }
        
        return None
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None

def main():
    print("="*80)
    print("DEEPFAKE AUDIO DETECTION MODEL FINDER & TESTER")
    print("="*80)
    
    # Install dependencies first
    check_and_install_dependencies()
    
    samples_dir = Path("samples")
    if not samples_dir.exists():
        print(f"❌ Samples directory not found: {samples_dir}")
        return
    
    # List of deepfake audio detection models to try
    # Prioritized by relevance to deepfake/spoofing detection
    models_to_try = [
        # Deepfake/Spoofing specific models
        ("microsoft/wavlm-base-plus", "audio_classification"),
        ("facebook/wav2vec2-base-960h", "audio_classification"),
        ("openai/whisper-base", "audio_classification"),
        ("superb/hubert-base-superb-er", "audio_classification"),  # Emotion recognition
        ("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition", "audio_classification"),
        # Alternative approaches
        ("facebook/wav2vec2-large-xlsr-53", "audio_classification"),
        ("nvidia/specter", "audio_classification"),
    ]
    
    results_list = []
    
    for model_name, model_type in models_to_try:
        result = test_model(model_name, samples_dir, model_type)
        if result:
            results_list.append(result)
    
    # Summary
    if results_list:
        print("\n" + "="*80)
        print("SUMMARY OF ALL TESTED MODELS")
        print("="*80)
        
        # Sort by accuracy
        results_list.sort(key=lambda x: x['accuracy'], reverse=True)
        
        for i, result in enumerate(results_list, 1):
            print(f"\n{i}. {result['model']}")
            print(f"   Overall Accuracy: {result['accuracy']:.2%}")
            print(f"   Real Audio: {result['real_accuracy']:.2%}")
            print(f"   Fake Audio: {result['fake_accuracy']:.2%}")
        
        if results_list:
            best = results_list[0]
            print("\n" + "="*80)
            print(f"✅ BEST MODEL: {best['model']}")
            print(f"   Accuracy: {best['accuracy']:.2%}")
            print("="*80)
    else:
        print("\n❌ No models could be successfully tested.")
        print("\nTroubleshooting:")
        print("  - Check that sample audio files exist in 'samples/' directory")
        print("  - Verify audio files are valid WAV format")
        print("  - Check internet connection for model downloads")

if __name__ == "__main__":
    main()
