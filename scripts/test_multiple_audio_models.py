"""
Try multiple HuggingFace audio deepfake detection models
Test with different models to find one that works
"""

import os
import sys
import torch
import librosa
from pathlib import Path

try:
    from transformers import AutoProcessor, AutoModelForAudioClassification, pipeline
except ImportError:
    print("Installing transformers...")
    os.system(f"{sys.executable} -m pip install transformers -q")
    from transformers import AutoProcessor, AutoModelForAudioClassification, pipeline

def test_audio_model(model_name, samples_dir):
    """Test a specific audio model"""
    print(f"\n{'=' * 80}")
    print(f"Testing Model: {model_name}")
    print(f"{'=' * 80}")
    
    try:
        print("📥 Downloading and loading model...")
        
        # Try using pipeline first (simpler API)
        try:
            classifier = pipeline("audio-classification", model=model_name)
            print("✅ Model loaded with pipeline API!")
            return test_with_pipeline(classifier, samples_dir, model_name)
        except Exception as e:
            print(f"⚠️  Pipeline failed: {e}")
            print("Trying direct model loading...")
            
            # Try direct loading
            processor = AutoProcessor.from_pretrained(model_name)
            model = AutoModelForAudioClassification.from_pretrained(model_name)
            print("✅ Model loaded directly!")
            return test_with_direct_model(processor, model, samples_dir, model_name)
            
    except Exception as e:
        print(f"❌ Failed to load {model_name}: {e}")
        return None

def test_with_pipeline(classifier, samples_dir, model_name):
    """Test using pipeline API - load audio with librosa first"""
    results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get audio files
    real_audio_files = sorted(list(samples_dir.glob("real_audio_*.wav")))[:5]
    fake_audio_files = sorted(list(samples_dir.glob("fake_audio_*.wav")))[:5]
    
    print("\nTesting REAL audio samples:")
    for audio_path in real_audio_files:
        try:
            # Load audio with librosa
            audio, sr = librosa.load(str(audio_path), sr=16000)
            
            # Run classifier with array instead of filename
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
            
            print(f"{correct} {audio_path.name:20s} | Predicted: {prediction:6s} ({confidence:.2%})")
        except Exception as e:
            print(f"❌ Error processing {audio_path.name}: {str(e)[:60]}")
    
    print("\nTesting FAKE audio samples:")
    for audio_path in fake_audio_files:
        try:
            # Load audio with librosa
            audio, sr = librosa.load(str(audio_path), sr=16000)
            
            # Run classifier with array instead of filename
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
            
            print(f"{correct} {audio_path.name:20s} | Predicted: {prediction:6s} ({confidence:.2%})")
        except Exception as e:
            print(f"❌ Error processing {audio_path.name}: {str(e)[:60]}")
    
    return calculate_metrics(results, model_name)

def test_with_direct_model(processor, model, samples_dir, model_name):
    """Test using direct model loading"""
    results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Get audio files
    real_audio_files = sorted(list(samples_dir.glob("real_audio_*.wav")))[:5]
    fake_audio_files = sorted(list(samples_dir.glob("fake_audio_*.wav")))[:5]
    
    print("\nTesting REAL audio samples:")
    for audio_path in real_audio_files:
        try:
            audio, sr = librosa.load(str(audio_path), sr=16000)
            inputs = processor(audio, sampling_rate=sr, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                predicted_class = torch.argmax(logits, dim=-1).item()
            
            labels = model.config.id2label
            prediction = labels[predicted_class].upper()
            confidence = probs[0, predicted_class].item()
            
            expected = "REAL"
            correct = "✅" if prediction == expected else "❌"
            
            results.append({
                'file': audio_path.name,
                'expected': expected,
                'predicted': prediction,
                'confidence': confidence,
                'correct': correct
            })
            
            print(f"{correct} {audio_path.name:20s} | Predicted: {prediction:6s} ({confidence:.2%})")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\nTesting FAKE audio samples:")
    for audio_path in fake_audio_files:
        try:
            audio, sr = librosa.load(str(audio_path), sr=16000)
            inputs = processor(audio, sampling_rate=sr, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                predicted_class = torch.argmax(logits, dim=-1).item()
            
            labels = model.config.id2label
            prediction = labels[predicted_class].upper()
            confidence = probs[0, predicted_class].item()
            
            expected = "FAKE"
            correct = "✅" if prediction == expected else "❌"
            
            results.append({
                'file': audio_path.name,
                'expected': expected,
                'predicted': prediction,
                'confidence': confidence,
                'correct': correct
            })
            
            print(f"{correct} {audio_path.name:20s} | Predicted: {prediction:6s} ({confidence:.2%})")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    return calculate_metrics(results, model_name)

def calculate_metrics(results, model_name):
    """Calculate and display metrics"""
    if not results:
        return None
    
    total = len(results)
    correct = sum(1 for r in results if r['correct'] == '✅')
    accuracy = correct / total
    
    real_results = [r for r in results if r['expected'] == 'REAL']
    fake_results = [r for r in results if r['expected'] == 'FAKE']
    
    real_correct = sum(1 for r in real_results if r['correct'] == '✅')
    fake_correct = sum(1 for r in fake_results if r['correct'] == '✅')
    
    real_accuracy = real_correct / len(real_results) if real_results else 0
    fake_accuracy = fake_correct / len(fake_results) if fake_results else 0
    
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Overall Accuracy: {accuracy:.2%}")
    print(f"Real Audio Accuracy: {real_accuracy:.2%}")
    print(f"Fake Audio Accuracy: {fake_accuracy:.2%}")
    
    return {
        'model': model_name,
        'accuracy': accuracy,
        'real_accuracy': real_accuracy,
        'fake_accuracy': fake_accuracy,
        'total_correct': correct,
        'total_samples': total
    }

def main():
    print("=" * 80)
    print("TESTING MULTIPLE HUGGINGFACE AUDIO DEEPFAKE DETECTION MODELS")
    print("=" * 80)
    
    samples_dir = Path("samples")
    if not samples_dir.exists():
        print(f"❌ Samples directory not found: {samples_dir}")
        return
    
    # List of models to try
    models = [
        # Audio classification models
        "facebook/wav2vec2-large-xlsr-53-english",
        "openai/whisper-base",
        "facebook/wav2vec2-base-960h",
        "microsoft/wavlm-base-plus",
        # Emotion/speaker detection (sometimes used for deepfake)
        "superb/hubert-base-superb-er",
        "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
    ]
    
    results_list = []
    
    for model_name in models:
        result = test_audio_model(model_name, samples_dir)
        if result:
            results_list.append(result)
    
    # Summary
    if results_list:
        print("\n" + "=" * 80)
        print("SUMMARY OF ALL MODELS")
        print("=" * 80)
        
        # Sort by accuracy
        results_list.sort(key=lambda x: x['accuracy'], reverse=True)
        
        for i, result in enumerate(results_list, 1):
            print(f"\n{i}. {result['model']}")
            print(f"   Overall Accuracy: {result['accuracy']:.2%}")
            print(f"   Real Audio: {result['real_accuracy']:.2%}")
            print(f"   Fake Audio: {result['fake_accuracy']:.2%}")
        
        best = results_list[0]
        print("\n" + "=" * 80)
        print(f"✅ BEST MODEL: {best['model']}")
        print(f"   Accuracy: {best['accuracy']:.2%}")
        print("=" * 80)
    else:
        print("\n❌ No models could be successfully loaded.")

if __name__ == "__main__":
    main()
