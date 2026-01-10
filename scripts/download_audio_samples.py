"""
Download/copy sample real and AI-generated audio files for testing
"""

import os
import shutil
from pathlib import Path
import requests

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
    print("DOWNLOADING SAMPLE AUDIO FILES FOR TESTING")
    print("=" * 80)
    print(f"Target directory: {samples_dir.absolute()}\n")
    
    # Sample audio files from public sources
    samples = {
        "real": [
            # Real human speech samples
            ("https://www2.cs.uic.edu/~i101/SoundFiles/preamble.wav", "real_speech1.wav"),
            ("https://www2.cs.uic.edu/~i101/SoundFiles/BabyElephantWalk60.wav", "real_music1.wav"),
        ]
    }
    
    print("📥 Downloading REAL audio samples...\n")
    for url, filename in samples["real"]:
        save_path = samples_dir / filename
        if save_path.exists():
            print(f"⏭️  Skipped (already exists): {filename}")
        else:
            download_file(url, save_path)
    
    # Copy samples from test dataset
    print("\n📋 Copying samples from test dataset...\n")
    
    test_real_dir = Path("data/audio/test/real")
    test_fake_dir = Path("data/audio/test/fake")
    
    if test_real_dir.exists():
        real_audios = list(test_real_dir.glob("*.wav"))[:5]
        for i, audio_path in enumerate(real_audios):
            dest = samples_dir / f"real_audio_{i+1}.wav"
            if not dest.exists():
                shutil.copy(audio_path, dest)
                print(f"✅ Copied: {dest.name}")
            else:
                print(f"⏭️  Skipped (already exists): {dest.name}")
    
    if test_fake_dir.exists():
        fake_audios = list(test_fake_dir.glob("*.wav"))[:5]
        for i, audio_path in enumerate(fake_audios):
            dest = samples_dir / f"fake_audio_{i+1}.wav"
            if not dest.exists():
                shutil.copy(audio_path, dest)
                print(f"✅ Copied: {dest.name}")
            else:
                print(f"⏭️  Skipped (already exists): {dest.name}")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    all_samples = list(samples_dir.glob("*.wav"))
    real_count = len([s for s in all_samples if 'real' in s.name.lower()])
    fake_count = len([s for s in all_samples if 'fake' in s.name.lower()])
    
    print(f"📁 Sample directory: {samples_dir.absolute()}")
    print(f"📊 Total audio samples: {len(all_samples)}")
    print(f"   - Real audio: {real_count}")
    print(f"   - Fake/AI audio: {fake_count}")
    print("\n✅ Audio samples ready for testing!")

if __name__ == "__main__":
    main()
