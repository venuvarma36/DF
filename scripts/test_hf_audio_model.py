"""
Download and test pretrained deepfake audio detection model from HuggingFace
Model: mo-thecreator/Deepfake-audio-detection
"""

import os
import sys
import torch
import librosa
from pathlib import Path

try:
    from transformers import AutoProcessor, AutoModelForAudioClassification
except ImportError:
    print("Installing transformers...")
    os.system(f"{sys.executable} -m pip install transformers -q")
    from transformers import AutoProcessor, AutoModelForAudioClassification

def download_and_test():
    print("=" * 80)
    print("DOWNLOADING PRETRAINED AUDIO DEEPFAKE DETECTION MODEL")
    print("=" * 80)
    
    # Try multiple models in order of preference
    models_to_try = [
        "speechbrain/ang-xlnet-base",
        "mo-thecreator/Deepfake-audio-detection"
    ]
    
    processor = None
    model = None
    model_name = None
    
    for candidate_model in models_to_try:
        try:
            print(f"\nTrying model: {candidate_model}")
            print("📥 Downloading processor...")
            processor = AutoProcessor.from_pretrained(candidate_model)
            print("✅ Processor downloaded!")
            
            print("📥 Downloading model...")
            model = AutoModelForAudioClassification.from_pretrained(candidate_model)
            print("✅ Model downloaded!")
            model_name = candidate_model
            break
        except Exception as e:
            print(f"⚠️  {candidate_model} failed: {e}")
            continue
    
    if model is None or processor is None:
        print("\n❌ Could not download any model. Trying simplified approach...")
        return False
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    print(f"\nDevice: {device}\n")
    
    # Test on samples
    print("=" * 80)
    print("TESTING ON AUDIO SAMPLES")
    print("=" * 80)
    
    samples_dir = Path("samples")
    
    if not samples_dir.exists():
        print(f"❌ Samples directory not found: {samples_dir}")
        return False
    
    # Get audio files
    audio_files = {
        "real": sorted(list(samples_dir.glob("real_*.wav"))),
        "fake": sorted(list(samples_dir.glob("fake_*.wav")))
    }
    
    results = []
    
    print("\n" + "=" * 80)
    print("TESTING REAL AUDIO SAMPLES")
    print("=" * 80)
    
    for audio_path in audio_files["real"]:
        try:
            # Load audio
            audio, sr = librosa.load(str(audio_path), sr=16000)
            
            # Preprocess
            inputs = processor(audio, sampling_rate=sr, return_tensors="pt").to(device)
            
            # Predict
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                predicted_class = torch.argmax(logits, dim=-1).item()
            
            # Get label
            labels = model.config.id2label
            prediction = labels[predicted_class]
            confidence = probs[0, predicted_class].item()
            
            expected = "real" if "real" in prediction.lower() else "fake"
            prediction_label = "REAL" if "real" in prediction.lower() else "FAKE"
            expected_label = "REAL"
            correct = "✅" if prediction_label == expected_label else "❌"
            
            results.append({
                'file': audio_path.name,
                'expected': expected_label,
                'predicted': prediction_label,
                'confidence': confidence,
                'raw_prediction': prediction,
                'correct': correct
            })
            
            print(f"{correct} {audio_path.name:25s} | "
                  f"Expected: {expected_label:4s} | "
                  f"Predicted: {prediction_label:4s} ({confidence:.2%}) | "
                  f"Raw: {prediction}")
            
        except Exception as e:
            print(f"❌ Error testing {audio_path.name}: {e}")
    
    print("\n" + "=" * 80)
    print("TESTING FAKE/AI AUDIO SAMPLES")
    print("=" * 80)
    
    for audio_path in audio_files["fake"]:
        try:
            # Load audio
            audio, sr = librosa.load(str(audio_path), sr=16000)
            
            # Preprocess
            inputs = processor(audio, sampling_rate=sr, return_tensors="pt").to(device)
            
            # Predict
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                predicted_class = torch.argmax(logits, dim=-1).item()
            
            # Get label
            labels = model.config.id2label
            prediction = labels[predicted_class]
            confidence = probs[0, predicted_class].item()
            
            expected = "real" if "real" in prediction.lower() else "fake"
            prediction_label = "REAL" if "real" in prediction.lower() else "FAKE"
            expected_label = "FAKE"
            correct = "✅" if prediction_label == expected_label else "❌"
            
            results.append({
                'file': audio_path.name,
                'expected': expected_label,
                'predicted': prediction_label,
                'confidence': confidence,
                'raw_prediction': prediction,
                'correct': correct
            })
            
            print(f"{correct} {audio_path.name:25s} | "
                  f"Expected: {expected_label:4s} | "
                  f"Predicted: {prediction_label:4s} ({confidence:.2%}) | "
                  f"Raw: {prediction}")
            
        except Exception as e:
            print(f"❌ Error testing {audio_path.name}: {e}")
    
    # Calculate metrics
    if results:
        print("\n" + "=" * 80)
        print("PERFORMANCE METRICS - PRETRAINED HF MODEL")
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
        
        print("=" * 80)
        print("COMPARISON WITH TRAINED MODEL")
        print("=" * 80)
        print(f"Trained Model Accuracy:     50.00% (0% real, 100% fake)")
        print(f"HuggingFace Model Accuracy: {accuracy:.2%} ({real_accuracy:.2%} real, {fake_accuracy:.2%} fake)")
        print(f"Improvement: {(accuracy - 0.5) * 100:+.2f}%")
        
        if accuracy >= 0.9:
            print("\n✅ EXCELLENT: Pretrained model performs very well!")
        elif accuracy >= 0.8:
            print("\n✅ VERY GOOD: Pretrained model is effective!")
        elif accuracy >= 0.7:
            print("\n✅ GOOD: Pretrained model performs well.")
        elif accuracy > 0.5:
            print("\n⚠️  FAIR: Pretrained model is better than trained model.")
        else:
            print("\n❌ POOR: Model needs improvement.")
        print("=" * 80)
        
        return True
    else:
        print("\n⚠️  No samples were successfully tested.")
        return False

if __name__ == "__main__":
    success = download_and_test()
    sys.exit(0 if success else 1)
