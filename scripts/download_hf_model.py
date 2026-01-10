"""
Download pretrained deepfake detection model from Hugging Face
Model: prithivMLmods/deepfake-detector-model-v1 (SigLIP-based)
"""

from transformers import AutoImageProcessor, SiglipForImageClassification
import torch
import os

def download_model():
    model_name = "prithivMLmods/deepfake-detector-model-v1"
    cache_dir = "checkpoints/pretrained_hf"
    
    print("=" * 80)
    print("DOWNLOADING PRETRAINED DEEPFAKE DETECTION MODEL")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Cache directory: {cache_dir}")
    print()
    
    # Create cache directory
    os.makedirs(cache_dir, exist_ok=True)
    
    try:
        # Download processor
        print("📥 Downloading image processor...")
        processor = AutoImageProcessor.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        print("✅ Image processor downloaded successfully!")
        
        # Download model
        print("\n📥 Downloading model (this may take a few minutes)...")
        model = SiglipForImageClassification.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        print("✅ Model downloaded successfully!")
        
        # Print model info
        print("\n" + "=" * 80)
        print("MODEL INFORMATION")
        print("=" * 80)
        print(f"Model type: {model.config.model_type}")
        print(f"Number of labels: {model.config.num_labels}")
        print(f"Image size: {model.config.vision_config.image_size}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Model size: ~{total_params * 4 / (1024**2):.1f} MB (fp32)")
        
        # Test the model
        print("\n" + "=" * 80)
        print("TESTING MODEL")
        print("=" * 80)
        print("Creating dummy input...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        
        # Use correct image size from config
        img_size = model.config.vision_config.image_size
        dummy_input = torch.randn(1, 3, img_size, img_size).to(device)
        with torch.no_grad():
            outputs = model(dummy_input)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            print(f"Input shape: {dummy_input.shape}")
            print(f"Output shape: {logits.shape}")
            print(f"Probabilities: Real={probs[0][0]:.4f}, Fake={probs[0][1]:.4f}")
        
        print("\n✅ Model is ready to use!")
        print(f"📁 Model cached at: {os.path.abspath(cache_dir)}")
        
        return processor, model
        
    except Exception as e:
        print(f"\n❌ Error downloading model: {e}")
        raise

if __name__ == "__main__":
    download_model()
