"""
Download pretrained audio deepfake detection models
Focuses on models specifically trained for fake audio detection
"""

import os
import sys
import torch
import requests
from pathlib import Path

def download_model_from_url(url, save_path, model_name):
    """Download model from direct URL"""
    print(f"\n{'='*80}")
    print(f"Downloading: {model_name}")
    print(f"{'='*80}")
    print(f"URL: {url}")
    print(f"Save path: {save_path}")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        downloaded = 0
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = downloaded / total_size * 100
                        print(f"\r  Progress: {progress:.1f}% ({downloaded/1024/1024:.1f} MB)", end='')
        
        print(f"\n✅ Downloaded: {save_path.name} ({save_path.stat().st_size/1024/1024:.2f} MB)")
        return True
    except Exception as e:
        print(f"\n❌ Failed to download: {e}")
        return False

def try_huggingface_model(model_id, save_dir):
    """Try to download a HuggingFace model"""
    print(f"\n{'='*80}")
    print(f"Trying HuggingFace Model: {model_id}")
    print(f"{'='*80}")
    
    try:
        # Install transformers if needed
        try:
            from transformers import AutoModel, AutoFeatureExtractor, AutoConfig
        except ImportError:
            print("Installing transformers...")
            os.system(f"{sys.executable} -m pip install transformers -q")
            from transformers import AutoModel, AutoFeatureExtractor, AutoConfig
        
        save_path = Path(save_dir) / model_id.replace('/', '_')
        save_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Downloading model to: {save_path}")
        
        # Download config first
        try:
            config = AutoConfig.from_pretrained(model_id)
            config.save_pretrained(save_path)
            print("✅ Config downloaded")
        except Exception as e:
            print(f"⚠️  Config download failed: {e}")
        
        # Download model
        try:
            model = AutoModel.from_pretrained(model_id)
            model.save_pretrained(save_path)
            print(f"✅ Model downloaded: {model_id}")
            
            # Get model size
            size_mb = sum(f.stat().st_size for f in save_path.rglob('*') if f.is_file()) / 1024 / 1024
            print(f"   Total size: {size_mb:.2f} MB")
            
            return True, save_path
        except Exception as e:
            print(f"❌ Model download failed: {e}")
            return False, None
            
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False, None

def download_torch_model(model_url, model_name, save_dir):
    """Download PyTorch model weights"""
    print(f"\n{'='*80}")
    print(f"Downloading PyTorch Model: {model_name}")
    print(f"{'='*80}")
    
    try:
        save_path = Path(save_dir) / f"{model_name}.pt"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Loading from: {model_url}")
        
        # Try to load from torch hub
        if "github.com" in model_url or "torch.hub" in model_url:
            try:
                checkpoint = torch.hub.load_state_dict_from_url(model_url, progress=True)
                torch.save(checkpoint, save_path)
                print(f"✅ Downloaded: {save_path.name} ({save_path.stat().st_size/1024/1024:.2f} MB)")
                return True, save_path
            except Exception as e:
                print(f"⚠️  Torch hub failed: {e}")
        
        # Try direct download
        return download_model_from_url(model_url, save_path, model_name), save_path if Path(save_path).exists() else None
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False, None

def main():
    print("="*80)
    print("DOWNLOAD PRETRAINED AUDIO DEEPFAKE DETECTION MODELS")
    print("="*80)
    
    # Create models directory
    models_dir = Path("checkpoints/pretrained_audio")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    downloaded_models = []
    
    # Option 1: Try specific audio deepfake detection models from HuggingFace
    hf_models = [
        # These are actual audio deepfake/spoofing detection models
        "Alibaba-DAMO-Academy/audio_deepfake_detection",
        "speechbrain/spkrec-ecapa-voxceleb",  # Speaker verification (can detect synthetic)
    ]
    
    print("\n" + "="*80)
    print("ATTEMPTING HUGGINGFACE MODELS")
    print("="*80)
    
    for model_id in hf_models:
        success, path = try_huggingface_model(model_id, models_dir)
        if success:
            downloaded_models.append({
                'name': model_id,
                'path': path,
                'type': 'huggingface'
            })
    
    # Option 2: Try downloading from GitHub releases or direct URLs
    # Note: These are example URLs - real pretrained models would need actual URLs
    print("\n" + "="*80)
    print("ATTEMPTING DIRECT DOWNLOADS")
    print("="*80)
    
    # Add any known direct download URLs here
    direct_models = [
        # Example: ("https://example.com/model.pt", "deepfake_detector_v1")
    ]
    
    for url, name in direct_models:
        success, path = download_torch_model(url, name, models_dir)
        if success:
            downloaded_models.append({
                'name': name,
                'path': path,
                'type': 'pytorch'
            })
    
    # Summary
    print("\n" + "="*80)
    print("DOWNLOAD SUMMARY")
    print("="*80)
    
    if downloaded_models:
        print(f"\n✅ Successfully downloaded {len(downloaded_models)} model(s):\n")
        for i, model in enumerate(downloaded_models, 1):
            print(f"{i}. {model['name']}")
            print(f"   Type: {model['type']}")
            print(f"   Path: {model['path']}")
            
            if Path(model['path']).exists():
                if Path(model['path']).is_dir():
                    size = sum(f.stat().st_size for f in Path(model['path']).rglob('*') if f.is_file()) / 1024 / 1024
                else:
                    size = Path(model['path']).stat().st_size / 1024 / 1024
                print(f"   Size: {size:.2f} MB")
            print()
    else:
        print("\n⚠️  No models were successfully downloaded.")
        print("\nREASONS:")
        print("  1. Many HuggingFace models require specific audio processing pipelines")
        print("  2. Most audio models are for speech recognition, not deepfake detection")
        print("  3. Deepfake-specific models may not be publicly available")
        print("\nRECOMMENDATIONS:")
        print("  1. Use the custom trained model (audio_best.pt)")
        print("  2. Retrain with balanced dataset for better performance")
        print("  3. Search for ASVspoof or voice anti-spoofing models")
        print("  4. Consider using the image deepfake model (works well at 70%)")
    
    print("="*80)

if __name__ == "__main__":
    main()
