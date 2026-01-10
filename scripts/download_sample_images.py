"""
Download sample real and fake images for testing
"""

import os
import requests
from pathlib import Path

def download_file(url, save_path):
    """Download a file from URL to save_path"""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"✅ Downloaded: {os.path.basename(save_path)}")
        return True
    except Exception as e:
        print(f"❌ Failed to download {url}: {e}")
        return False

def main():
    # Create samples directory
    samples_dir = Path("samples")
    samples_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("DOWNLOADING SAMPLE IMAGES FOR TESTING")
    print("=" * 80)
    print(f"Target directory: {samples_dir.absolute()}\n")
    
    # Sample images - using publicly available test images
    samples = {
        "real": [
            # Real face images from public datasets
            ("https://raw.githubusercontent.com/pytorch/vision/main/gallery/assets/dog1.jpg", "real_dog.jpg"),
            ("https://raw.githubusercontent.com/pytorch/vision/main/gallery/assets/dog2.jpg", "real_dog2.jpg"),
            ("https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/biden.jpg", "real_person1.jpg"),
            ("https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/obama.jpg", "real_person2.jpg"),
            ("https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/obama2.jpg", "real_person3.jpg"),
            ("https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/obama-240p.jpg", "real_person4.jpg"),
            ("https://raw.githubusercontent.com/opencv/opencv/master/samples/data/messi5.jpg", "real_person5.jpg"),
            ("https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg", "real_person6.jpg"),
        ],
        "fake": [
            # Note: For actual deepfake samples, you would need access to deepfake datasets
            # These are placeholders - will copy from your test data instead
        ]
    }
    
    print("📥 Downloading REAL sample images...\n")
    for url, filename in samples["real"]:
        save_path = samples_dir / filename
        if save_path.exists():
            print(f"⏭️  Skipped (already exists): {filename}")
        else:
            download_file(url, save_path)
    
    # Copy some samples from test data if available
    print("\n📋 Copying samples from test dataset...\n")
    
    test_real_dir = Path("data/image/test/real")
    test_fake_dir = Path("data/image/test/fake")
    
    if test_real_dir.exists():
        real_images = list(test_real_dir.glob("*.*"))[:10]
        for i, img_path in enumerate(real_images):
            dest = samples_dir / f"real_sample_{i+1}{img_path.suffix}"
            if not dest.exists():
                import shutil
                shutil.copy(img_path, dest)
                print(f"✅ Copied: {dest.name}")
            else:
                print(f"⏭️  Skipped (already exists): {dest.name}")
    
    if test_fake_dir.exists():
        fake_images = list(test_fake_dir.glob("*.*"))[:10]
        for i, img_path in enumerate(fake_images):
            dest = samples_dir / f"fake_sample_{i+1}{img_path.suffix}"
            if not dest.exists():
                import shutil
                shutil.copy(img_path, dest)
                print(f"✅ Copied: {dest.name}")
            else:
                print(f"⏭️  Skipped (already exists): {dest.name}")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    all_samples = list(samples_dir.glob("*.*"))
    real_count = len([s for s in all_samples if 'real' in s.name.lower()])
    fake_count = len([s for s in all_samples if 'fake' in s.name.lower()])
    
    print(f"📁 Sample directory: {samples_dir.absolute()}")
    print(f"📊 Total samples: {len(all_samples)}")
    print(f"   - Real images: {real_count}")
    print(f"   - Fake images: {fake_count}")
    print("\n✅ Sample images ready for testing!")
    print(f"\nTest them with: python scripts/test_hf_model.py")

if __name__ == "__main__":
    main()
