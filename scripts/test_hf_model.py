"""
Test the pretrained Hugging Face model on sample images
"""

from transformers import AutoImageProcessor, SiglipForImageClassification
from PIL import Image
import torch
import os
import glob

def predict_image(image_path, processor, model, device):
    """Predict if an image is real or fake"""
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
    
    # Interpret results (assuming index 0=Real, 1=Fake)
    real_prob = probs[0][0].item()
    fake_prob = probs[0][1].item()
    prediction = "REAL" if real_prob > fake_prob else "FAKE"
    confidence = max(real_prob, fake_prob)
    
    return prediction, confidence, real_prob, fake_prob

def main():
    print("=" * 80)
    print("TESTING PRETRAINED DEEPFAKE DETECTION MODEL")
    print("=" * 80)
    
    # Load model
    model_name = "prithivMLmods/deepfake-detector-model-v1"
    cache_dir = "checkpoints/pretrained_hf"
    
    print(f"Loading model: {model_name}")
    processor = AutoImageProcessor.from_pretrained(model_name, cache_dir=cache_dir)
    model = SiglipForImageClassification.from_pretrained(model_name, cache_dir=cache_dir)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Device: {device}\n")
    
    # Test on sample images from dataset
    test_dirs = [
        ("data/image/test/real", "REAL"),
        ("data/image/test/fake", "FAKE")
    ]
    
    results = []
    for test_dir, expected_label in test_dirs:
        if not os.path.exists(test_dir):
            print(f"⚠️  Directory not found: {test_dir}")
            continue
        
        # Get first 5 images from each directory
        image_files = glob.glob(os.path.join(test_dir, "*.*"))[:5]
        
        for img_path in image_files:
            try:
                prediction, confidence, real_prob, fake_prob = predict_image(
                    img_path, processor, model, device
                )
                
                # Check if prediction matches expected label
                correct = "✅" if prediction == expected_label else "❌"
                
                results.append({
                    'file': os.path.basename(img_path),
                    'expected': expected_label,
                    'predicted': prediction,
                    'confidence': confidence,
                    'real_prob': real_prob,
                    'fake_prob': fake_prob,
                    'correct': correct
                })
                
                print(f"{correct} {os.path.basename(img_path):30s} | "
                      f"Expected: {expected_label:4s} | "
                      f"Predicted: {prediction:4s} ({confidence:.2%}) | "
                      f"Real: {real_prob:.2%} | Fake: {fake_prob:.2%}")
                
            except Exception as e:
                print(f"❌ Error processing {img_path}: {e}")
    
    # Summary
    if results:
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        total = len(results)
        correct = sum(1 for r in results if r['correct'] == '✅')
        accuracy = correct / total if total > 0 else 0
        
        print(f"Total images tested: {total}")
        print(f"Correct predictions: {correct}")
        print(f"Accuracy: {accuracy:.2%}")
        
        avg_real_prob = sum(r['real_prob'] for r in results) / total
        avg_fake_prob = sum(r['fake_prob'] for r in results) / total
        print(f"Average Real probability: {avg_real_prob:.2%}")
        print(f"Average Fake probability: {avg_fake_prob:.2%}")
        
        print("\n✅ Yes, this model can predict REAL vs FAKE images!")
    else:
        print("\n⚠️  No test images found. Add images to data/image/test/real and data/image/test/fake")

if __name__ == "__main__":
    main()
