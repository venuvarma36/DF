"""
Organize Kaggle In-The-Wild Audio Deepfake dataset into train/val/test splits
"""
import zipfile
import os
import shutil
from pathlib import Path
import random
from sklearn.model_selection import train_test_split

# Configuration
DOWNLOAD_PATH = os.path.expanduser(r"~\Downloads\release_in_the_wild.zip")
AUDIO_DATA_PATH = r"C:\DeepFake_detection\data\audio"
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

print("=" * 80)
print("KAGGLE AUDIO DEEPFAKE DATASET ORGANIZATION")
print("=" * 80)

# Check if zip exists
if not os.path.exists(DOWNLOAD_PATH):
    print(f"❌ Zip file not found at: {DOWNLOAD_PATH}")
    exit(1)

print(f"✓ Found dataset: {DOWNLOAD_PATH}")
print(f"  Size: {os.path.getsize(DOWNLOAD_PATH) / (1024**3):.2f} GB")

# Create backup of existing data if it exists
backup_dir = os.path.join(AUDIO_DATA_PATH, "backup_old")
if os.path.exists(os.path.join(AUDIO_DATA_PATH, "fake")):
    print(f"\n⚠️  Backing up existing data to: {backup_dir}")
    os.makedirs(backup_dir, exist_ok=True)
    if os.path.exists(os.path.join(AUDIO_DATA_PATH, "fake")):
        shutil.move(os.path.join(AUDIO_DATA_PATH, "fake"), os.path.join(backup_dir, "fake"))
    if os.path.exists(os.path.join(AUDIO_DATA_PATH, "real")):
        shutil.move(os.path.join(AUDIO_DATA_PATH, "real"), os.path.join(backup_dir, "real"))
    print("✓ Backup complete")

# Extract zip
print("\n📦 Extracting dataset...")
extract_dir = os.path.join(AUDIO_DATA_PATH, "extracted")
os.makedirs(extract_dir, exist_ok=True)

with zipfile.ZipFile(DOWNLOAD_PATH, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)
print("✓ Extraction complete")

# Get file lists
fake_dir = os.path.join(extract_dir, "fake")
real_dir = os.path.join(extract_dir, "real")

fake_files = [f for f in os.listdir(fake_dir) if f.endswith('.wav')]
real_files = [f for f in os.listdir(real_dir) if f.endswith('.wav')]

print(f"\n📊 Dataset Summary:")
print(f"  Fake files: {len(fake_files)}")
print(f"  Real files: {len(real_files)}")
print(f"  Total: {len(fake_files) + len(real_files)}")

# Create train/val/test directories
for split in ['train', 'val', 'test']:
    for label in ['fake', 'real']:
        os.makedirs(os.path.join(AUDIO_DATA_PATH, split, label), exist_ok=True)

# Split files with proper seeding
random.seed(42)
fake_train, fake_temp = train_test_split(
    fake_files, test_size=(1 - TRAIN_RATIO), random_state=42
)
fake_val, fake_test = train_test_split(
    fake_temp, 
    test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO), 
    random_state=42
)

real_train, real_temp = train_test_split(
    real_files, test_size=(1 - TRAIN_RATIO), random_state=42
)
real_val, real_test = train_test_split(
    real_temp, 
    test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO), 
    random_state=42
)

print(f"\n📁 Split Distribution (70% train, 15% val, 15% test):")
print(f"  FAKE  - Train: {len(fake_train)}, Val: {len(fake_val)}, Test: {len(fake_test)}")
print(f"  REAL  - Train: {len(real_train)}, Val: {len(real_val)}, Test: {len(real_test)}")

# Copy files
def copy_files(file_list, src_dir, dst_dir, label):
    """Copy files from source to destination with progress"""
    total = len(file_list)
    for idx, filename in enumerate(file_list):
        src = os.path.join(src_dir, filename)
        dst = os.path.join(dst_dir, filename)
        shutil.copy2(src, dst)
        if (idx + 1) % max(1, total // 10) == 0:
            print(f"    {label}: {idx + 1}/{total} copied", end='\r')
    print(f"    {label}: {total}/{total} copied ✓")

print("\n📋 Copying files...")

print("  FAKE:")
copy_files(fake_train, fake_dir, os.path.join(AUDIO_DATA_PATH, "train", "fake"), "train")
copy_files(fake_val, fake_dir, os.path.join(AUDIO_DATA_PATH, "val", "fake"), "val")
copy_files(fake_test, fake_dir, os.path.join(AUDIO_DATA_PATH, "test", "fake"), "test")

print("  REAL:")
copy_files(real_train, real_dir, os.path.join(AUDIO_DATA_PATH, "train", "real"), "train")
copy_files(real_val, real_dir, os.path.join(AUDIO_DATA_PATH, "val", "real"), "val")
copy_files(real_test, real_dir, os.path.join(AUDIO_DATA_PATH, "test", "real"), "test")

print("✓ All files copied")

# Cleanup extracted directory
print("\n🧹 Cleaning up temporary files...")
shutil.rmtree(extract_dir)
print("✓ Cleanup complete")

# Verify final structure
print("\n✅ FINAL STRUCTURE:")
print(f"  data/audio/")
for split in ['train', 'val', 'test']:
    fake_count = len(os.listdir(os.path.join(AUDIO_DATA_PATH, split, 'fake')))
    real_count = len(os.listdir(os.path.join(AUDIO_DATA_PATH, split, 'real')))
    total = fake_count + real_count
    print(f"    {split}/")
    print(f"      fake/  {fake_count} files")
    print(f"      real/  {real_count} files")
    print(f"      Total: {total} files")

total_files = sum([
    len(os.listdir(os.path.join(AUDIO_DATA_PATH, split, label)))
    for split in ['train', 'val', 'test']
    for label in ['fake', 'real']
])

print(f"\n🎉 Organization Complete!")
print(f"   Total audio files: {total_files}")
print(f"   Location: {AUDIO_DATA_PATH}")
