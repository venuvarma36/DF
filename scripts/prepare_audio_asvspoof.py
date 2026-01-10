"""
Download and prepare ASVspoof 2021 DF (Deepfake) track dataset
"""
import os
import requests
import tarfile
import shutil
from pathlib import Path
from tqdm import tqdm
import pandas as pd

# ASVspoof 2021 DF track URLs
ASVSPOOF_BASE = "https://datashare.ed.ac.uk/bitstream/handle/10283/3974"
DATASETS = {
    "train": f"{ASVSPOOF_BASE}/ASVspoof2021_DF_eval_part00.tar.gz",
    "keys": "https://www.asvspoof.org/resources/DF-keys-full.tar.gz",
}


def download_file(url, destination):
    """Download file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as f, tqdm(
        desc=destination.name,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))


def download_asvspoof():
    """Download ASVspoof 2021 dataset"""
    print("=" * 60)
    print("ASVspoof 2021 DF Dataset Download")
    print("=" * 60)
    
    raw_dir = Path("data/raw/asvspoof2021")
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n📥 Downloading ASVspoof 2021 DF track...")
    print("Note: Dataset is ~13GB. This may take a while.")
    print("\nOfficial download page:")
    print("https://www.asvspoof.org/index2021.html")
    
    # Download training data
    print("\n1️⃣ Downloading training data...")
    train_tar = raw_dir / "train.tar.gz"
    
    if not train_tar.exists():
        print("\n⚠️ Automatic download may not work due to website restrictions.")
        print("\nPlease manually download from:")
        print("https://www.asvspoof.org/index2021.html")
        print("  - ASVspoof2021_DF_eval (train)")
        print("  - Protocol files")
        print(f"\nPlace files in: {raw_dir.absolute()}")
        
        response = input("\nHave you downloaded the files manually? (y/n): ")
        if response.lower() != 'y':
            return False
    
    print("\n✅ Files ready for extraction")
    return True


def extract_archives():
    """Extract downloaded archives"""
    print("\n" + "=" * 60)
    print("Extracting archives...")
    print("=" * 60)
    
    raw_dir = Path("data/raw/asvspoof2021")
    
    for tar_file in raw_dir.glob("*.tar.gz"):
        print(f"\n📦 Extracting {tar_file.name}...")
        with tarfile.open(tar_file, 'r:gz') as tar:
            tar.extractall(raw_dir)
    
    print("\n✅ Extraction complete")


def organize_dataset():
    """Organize ASVspoof files into train/val/test structure"""
    print("\n" + "=" * 60)
    print("Organizing dataset...")
    print("=" * 60)
    
    raw_dir = Path("data/raw/asvspoof2021")
    target_dir = Path("data/audio")
    
    # Read protocol file to get labels
    protocol_file = None
    for pf in raw_dir.rglob("*protocol*.txt"):
        protocol_file = pf
        break
    
    if not protocol_file:
        print("⚠️ Protocol file not found. Using directory structure.")
        # Fallback: organize by directory
        organize_by_directory(raw_dir, target_dir)
        return
    
    print(f"📄 Reading protocol: {protocol_file.name}")
    
    # Parse protocol file
    # Format: speaker file system - label
    protocol = []
    with open(protocol_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                speaker, utt, _, attack, label = parts[:5]
                is_bonafide = (label == 'bonafide')
                protocol.append({
                    'file': utt,
                    'label': 'real' if is_bonafide else 'fake',
                    'speaker': speaker,
                })
    
    print(f"Found {len(protocol)} utterances")
    
    # Split: 70% train, 15% val, 15% test
    import random
    random.seed(42)
    random.shuffle(protocol)
    
    n_train = int(len(protocol) * 0.7)
    n_val = int(len(protocol) * 0.15)
    
    splits = {
        'train': protocol[:n_train],
        'val': protocol[n_train:n_train + n_val],
        'test': protocol[n_train + n_val:],
    }
    
    # Find audio files and copy
    audio_files = list(raw_dir.rglob("*.flac")) + list(raw_dir.rglob("*.wav"))
    audio_map = {f.stem: f for f in audio_files}
    
    for split, items in splits.items():
        for item in tqdm(items, desc=f"Copying {split}"):
            file_stem = item['file'].replace('.flac', '').replace('.wav', '')
            
            if file_stem in audio_map:
                src = audio_map[file_stem]
                dst_dir = target_dir / split / item['label']
                dst_dir.mkdir(parents=True, exist_ok=True)
                
                dst = dst_dir / src.name
                if not dst.exists():
                    shutil.copy2(src, dst)
    
    # Print statistics
    print("\n" + "=" * 60)
    print("Dataset Statistics:")
    print("=" * 60)
    for split in ['train', 'val', 'test']:
        for label in ['real', 'fake']:
            label_dir = target_dir / split / label
            if label_dir.exists():
                count = len(list(label_dir.glob("*")))
                print(f"{split}/{label}: {count} files")


def organize_by_directory(raw_dir, target_dir):
    """Fallback: organize by directory structure"""
    print("\n📁 Organizing by directory structure...")
    
    audio_files = list(raw_dir.rglob("*.flac")) + list(raw_dir.rglob("*.wav"))
    
    # Heuristic: files with 'bonafide' or in bonafide dir = real
    import random
    random.seed(42)
    random.shuffle(audio_files)
    
    n_train = int(len(audio_files) * 0.7)
    n_val = int(len(audio_files) * 0.15)
    
    splits = {
        'train': audio_files[:n_train],
        'val': audio_files[n_train:n_train + n_val],
        'test': audio_files[n_train + n_val:],
    }
    
    for split, files in splits.items():
        for f in tqdm(files, desc=f"Copying {split}"):
            # Determine label from path
            label = 'real' if 'bonafide' in str(f).lower() else 'fake'
            
            dst_dir = target_dir / split / label
            dst_dir.mkdir(parents=True, exist_ok=True)
            
            dst = dst_dir / f.name
            if not dst.exists():
                shutil.copy2(f, dst)


def main():
    print("\n🎤 ASVspoof 2021 Dataset Preparation Pipeline")
    print("=" * 60)
    
    # Step 1: Download
    print("\n📥 Step 1/3: Downloading ASVspoof 2021...")
    if not download_asvspoof():
        print("\n⚠️ Please download manually and rerun this script")
        return
    
    # Step 2: Extract
    print("\n📦 Step 2/3: Extracting archives...")
    response = input("\nProceed with extraction? (y/n): ")
    if response.lower() == 'y':
        extract_archives()
    
    # Step 3: Organize
    print("\n📂 Step 3/3: Organizing into train/val/test...")
    response = input("\nProceed with organization? (y/n): ")
    if response.lower() == 'y':
        organize_dataset()
    
    print("\n" + "=" * 60)
    print("✅ ASVspoof 2021 dataset preparation complete!")
    print("=" * 60)
    print("\nDataset location: data/audio/")
    print("\nReady to train! Run:")
    print("  python -m src.training.train_audio")


if __name__ == "__main__":
    main()
