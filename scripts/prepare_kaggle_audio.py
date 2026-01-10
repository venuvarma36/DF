import os
import shutil
import random
from pathlib import Path
import sys

def scan_for_folders(root_dir):
    """Recursively find folders named 'real' and 'fake' (case insensitive)."""
    real_dirs = []
    fake_dirs = []
    
    print(f"Scanning {root_dir}...")
    root = Path(root_dir)
    
    for path in root.rglob("*"):
        if path.is_dir():
            if path.name.lower() == 'real':
                real_dirs.append(path)
            elif path.name.lower() == 'fake':
                fake_dirs.append(path)
                
    return real_dirs, fake_dirs

def get_audio_files(dirs):
    files = []
    for d in dirs:
        for f in d.rglob("*"):
            if f.suffix.lower() in ['.wav', '.mp3', '.flac', '.m4a']:
                files.append(f)
    return files

def main():
    print("="*60)
    print("AUDIO DATASET PREPARATION")
    print("="*60)
    
    # Configuration
    source_root = Path("data/raw/extracted")
    target_root = Path("data/audio")
    EXTENSIONS = {'.wav', '.mp3', '.flac', '.m4a'}
    
    # 1. Locate Data
    if not source_root.exists():
        print(f"❌ Source directory not found: {source_root}")
        print("   Please wait for archive extraction to complete, or check paths.")
        return

    print("🔍 Searching for 'REAL' and 'FAKE' audio folders...")
    real_dirs, fake_dirs = scan_for_folders(source_root)
    
    print(f"   Found {len(real_dirs)} 'real' folders: {[str(d) for d in real_dirs]}")
    print(f"   Found {len(fake_dirs)} 'fake' folders: {[str(d) for d in fake_dirs]}")
    
    if not real_dirs or not fake_dirs:
        print("❌ Could not find both 'real' and 'fake' folders.")
        return

    # 2. Collect Files
    real_files = get_audio_files(real_dirs)
    fake_files = get_audio_files(fake_dirs)
    
    print(f"\n📊 Found files:")
    print(f"   Real: {len(real_files)}")
    print(f"   Fake: {len(fake_files)}")
    
    if len(real_files) == 0 or len(fake_files) == 0:
        print("❌ No audio files found in directories.")
        return

    # 3. Setup Destination
    if target_root.exists():
        # Check if empty (ignoring .gitkeep or similar small files)
        has_content = any(p.name not in ['.gitkeep'] for p in target_root.rglob('*') if p.is_file())
        if has_content:
            print(f"\n⚠️  Target directory {target_root} is not empty.")
            # Simple interaction simulation or auto-decision
            # For this task, we will clear 'train', 'val', 'test' subdirs but keep others
            print("   Cleaning up old train/val/test data...")
            for split in ['train', 'val', 'test']:
                p = target_root / split
                if p.exists():
                    shutil.rmtree(p)

    # 4. Distribute Data (80/10/10)
    random.seed(42)
    random.shuffle(real_files)
    random.shuffle(fake_files)
    
    splits = {
        'train': (0.0, 0.8),
        'val':   (0.8, 0.9),
        'test':  (0.9, 1.0)
    }
    
    print("\n🚚 Processing and chunking files...")
    
    CHUNK_DURATION = 4.0 # seconds
    
    import soundfile as sf
    import math
    import numpy as np

    total_chunks = 0
    
    for label_name, file_list in [('real', real_files), ('fake', fake_files)]:
        total_files = len(file_list)
        for split_name, (start_pct, end_pct) in splits.items():
            start_idx = int(total_files * start_pct)
            end_idx = int(total_files * end_pct)
            subset = file_list[start_idx:end_idx]
            
            dest_dir = target_root / split_name / label_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"   -> {split_name}/{label_name}: processing {len(subset)} files...")
            
            for src in subset:
                try:
                    # Load audio
                    # Read info first
                    info = sf.info(src)
                    sr = info.samplerate
                    chunk_frames = int(CHUNK_DURATION * sr)
                    
                    # Process in blocks to avoid huge RAM (though 80MB is fine, blocks are safer)
                    # For simplicity/speed with soundfile, we can read blocks.
                    
                    blocks = sf.blocks(src, blocksize=chunk_frames)
                    
                    for i, block in enumerate(blocks):
                        # Ensure block is long enough (at least 1 sec)
                        if len(block) < sr * 1.0:
                            continue
                            
                        # If stereo, mix to mono
                        if block.ndim > 1:
                            block = np.mean(block, axis=1)
                            
                        chunk_name = f"{src.stem}_chunk_{i:04d}.wav"
                        dst = dest_dir / chunk_name
                        
                        sf.write(dst, block, sr)
                        total_chunks += 1
                        
                except Exception as e:
                    print(f"      ⚠️ Failed to process {src.name}: {e}")

    print(f"\n✅ Dataset processing complete! Generated {total_chunks} chunks.")
    
    # 5. Check
    total_dest = len(list(target_root.rglob("*.wav")))
    print(f"   Total files in {target_root}: {total_dest}")

if __name__ == "__main__":
    main()
