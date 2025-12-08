"""
Organize Video Frames for Training
Works with extracted frames from data/frames/
"""

import shutil
from pathlib import Path
import random
from tqdm import tqdm


def organize_frames(
    frames_dir="data/frames",
    output_dir="data",
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15
):
    """Organize extracted frames into train/val/test splits"""
    
    print("="*70)
    print("BlueDepth-Crescent Frame Organization")
    print("="*70)
    
    frames_dir = Path(frames_dir)
    output_dir = Path(output_dir)
    
    # Collect frames
    print("\n[1] Collecting frames...")
    all_frames = []
    
    for video_folder in frames_dir.iterdir():
        if video_folder.is_dir():
            frames = list(video_folder.glob("*.jpg")) + list(video_folder.glob("*.png"))
            all_frames.extend(frames)
            print(f"   {video_folder.name}: {len(frames)} frames")
    
    if not all_frames:
        print(f"\n❌ No frames found in {frames_dir}")
        print("\n💡 Extract frames first:")
        print("   python scripts/extract_video_frames.py")
        return
    
    print(f"\n✅ Total: {len(all_frames)} frames")
    
    # Shuffle
    random.seed(42)
    random.shuffle(all_frames)
    
    # Split
    total = len(all_frames)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    train_frames = all_frames[:train_end]
    val_frames = all_frames[train_end:val_end]
    test_frames = all_frames[val_end:]
    
    print(f"\n[2] Split:")
    print(f"   Train: {len(train_frames)} ({len(train_frames)/total*100:.1f}%)")
    print(f"   Val:   {len(val_frames)} ({len(val_frames)/total*100:.1f}%)")
    print(f"   Test:  {len(test_frames)} ({len(test_frames)/total*100:.1f}%)")
    
    # Create directories
    train_dir = output_dir / "train" / "hazy"
    val_dir = output_dir / "val" / "hazy"
    test_dir = output_dir / "test" / "hazy"
    
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy files
    print("\n[3] Organizing...")
    
    for idx, frame in enumerate(tqdm(train_frames, desc="  Train")):
        shutil.copy2(frame, train_dir / f"train_{idx:05d}{frame.suffix}")
    
    for idx, frame in enumerate(tqdm(val_frames, desc="  Val")):
        shutil.copy2(frame, val_dir / f"val_{idx:05d}{frame.suffix}")
    
    for idx, frame in enumerate(tqdm(test_frames, desc="  Test")):
        shutil.copy2(frame, test_dir / f"test_{idx:05d}{frame.suffix}")
    
    print("\n" + "="*70)
    print("✅ Organization Complete!")
    print("="*70)
    
    print(f"\n📂 Structure:")
    print(f"   {output_dir}/")
    print(f"   ├── train/hazy/ ({len(train_frames)} frames)")
    print(f"   ├── val/hazy/   ({len(val_frames)} frames)")
    print(f"   └── test/hazy/  ({len(test_frames)} frames)")
    
    print(f"\n🚀 Start training:")
    print(f"   python main.py train --model standard --epochs 100")


if __name__ == "__main__":
    organize_frames()
