"""
Quick dataset downloader - simplified version
Uses wget/curl for direct downloads where possible
"""
import os
import sys
import subprocess
from pathlib import Path


def run_cmd(cmd):
    """Run shell command"""
    print(f"\n▶ {cmd}")
    result = subprocess.run(cmd, shell=True)
    return result.returncode == 0


def setup_directories():
    """Create necessary directories"""
    dirs = [
        "data/raw/dfdc",
        "data/raw/asvspoof2021",
        "data/raw/deeperforensics",
        "data/raw/speechfake",
        "data/audio/train/real",
        "data/audio/train/fake",
        "data/audio/val/real",
        "data/audio/val/fake",
        "data/audio/test/real",
        "data/audio/test/fake",
        "data/image/train/real",
        "data/image/train/fake",
        "data/image/val/real",
        "data/image/val/fake",
        "data/image/test/real",
        "data/image/test/fake",
        "logs",
        "checkpoints",
    ]
    
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    
    print("✅ Directory structure created")


def main():
    print("\n" + "="*60)
    print("📥 DATASET DOWNLOAD GUIDE")
    print("="*60)
    
    setup_directories()
    
    print("\n" + "="*60)
    print("IMAGE DATASETS")
    print("="*60)
    
    print("\n1️⃣ DFDC (Deepfake Detection Challenge)")
    print("   Size: ~470GB (full), ~100GB (subset recommended)")
    print("   Source: https://www.kaggle.com/c/deepfake-detection-challenge")
    print("\n   Setup:")
    print("   - Install Kaggle: pip install kaggle")
    print("   - Get API token from: https://www.kaggle.com/settings")
    print("   - Place kaggle.json in: ~/.kaggle/ (Linux/Mac) or C:\\Users\\<You>\\.kaggle\\ (Windows)")
    print("   - Accept competition rules at the URL above")
    print("\n   Download:")
    print("   kaggle competitions download -c deepfake-detection-challenge")
    print("   kaggle competitions download -c deepfake-detection-challenge -f dfdc_train_part_0.zip")
    print("   (Download parts 0-9 for ~100GB subset)")
    
    print("\n2️⃣ DeeperForensics (Optional)")
    print("   Size: ~100GB")
    print("   Source: https://github.com/EndlessSora/DeeperForensics-1.0")
    print("   Manual download required from project page")
    
    print("\n" + "="*60)
    print("AUDIO DATASETS")
    print("="*60)
    
    print("\n3️⃣ ASVspoof 2021 DF Track")
    print("   Size: ~13GB")
    print("   Source: https://www.asvspoof.org/index2021.html")
    print("\n   Download:")
    print("   - Visit: https://www.asvspoof.org/index2021.html")
    print("   - Register and download DF track (Deepfake)")
    print("   - Place files in: data/raw/asvspoof2021/")
    
    print("\n4️⃣ SpeechFake (Optional)")
    print("   Size: ~5GB")
    print("   Source: https://github.com/Edresson/SpeechFake")
    print("   Manual download from GitHub releases")
    
    print("\n" + "="*60)
    print("AUTOMATED PREPARATION")
    print("="*60)
    print("\nAfter downloading, run these scripts to prepare data:")
    print("\n  For images:")
    print("    python scripts/prepare_image_dfdc.py")
    print("\n  For audio:")
    print("    python scripts/prepare_audio_asvspoof.py")
    print("\n  Or use the automated pipeline:")
    print("    python scripts/train_pipeline.py")
    
    print("\n" + "="*60)
    print("QUICK START (Smaller Dataset)")
    print("="*60)
    print("\nFor testing, download smaller subsets:")
    print("  - DFDC: First 2 parts only (~20GB) = ~40K frames")
    print("  - ASVspoof: Training set only (~10GB)")
    print("\nThis gives you ~10K images and ~50K audio samples - enough to start!")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
