#!/usr/bin/env python3
"""
Create Validation Split from Training Data
Splits training data: 80% train, 20% validation
"""

import shutil
from pathlib import Path
import random
import argparse
from datetime import datetime  # ← FIX: Import at top, not inside f-string

def create_validation_split(
    train_dir: str = "data/train",
    val_dir: str = "data/val",
    val_ratio: float = 0.2,
    seed: int = 42
):
    """
    Split training data into train and validation sets
    
    Args:
        train_dir: Path to training directory
        val_dir: Path to validation directory (will be created)
        val_ratio: Fraction for validation (0.2 = 20%)
        seed: Random seed for reproducibility
    """
    
    random.seed(seed)
    
    train_path = Path(train_dir)
    val_path = Path(val_dir)
    
    # Create validation directories
    val_hazy = val_path / "hazy"
    val_clear = val_path / "clear"
    val_hazy.mkdir(parents=True, exist_ok=True)
    val_clear.mkdir(parents=True, exist_ok=True)
    
    # Get all training images
    train_hazy = train_path / "hazy"
    train_clear = train_path / "clear"
    
    hazy_images = sorted(list(train_hazy.glob("*.jpg")))
    
    print(f"Found {len(hazy_images)} training image pairs")
    
    # Calculate validation split
    num_val = int(len(hazy_images) * val_ratio)
    print(f"Creating validation split: {num_val} images ({val_ratio*100:.0f}%)")
    
    # Randomly sample validation indices
    val_indices = random.sample(range(len(hazy_images)), num_val)
    val_filenames = [hazy_images[i].name for i in val_indices]
    
    # Move images to validation folder
    moved_count = 0
    for filename in val_filenames:
        # Move hazy image
        src_hazy = train_hazy / filename
        dst_hazy = val_hazy / filename
        if src_hazy.exists():
            shutil.move(str(src_hazy), str(dst_hazy))
            moved_count += 1
        
        # Move clear image
        src_clear = train_clear / filename
        dst_clear = val_clear / filename
        if src_clear.exists():
            shutil.move(str(src_clear), str(dst_clear))
    
    print(f"✓ Moved {moved_count} image pairs to validation")
    
    # Count remaining images
    train_remaining = len(list(train_hazy.glob("*.jpg")))
    val_total = len(list(val_hazy.glob("*.jpg")))
    
    print(f"\nFinal split:")
    print(f"  Training:   {train_remaining} pairs ({train_remaining/(train_remaining+val_total)*100:.1f}%)")
    print(f"  Validation: {val_total} pairs ({val_total/(train_remaining+val_total)*100:.1f}%)")
    print(f"  Testing:    920 pairs (unchanged)")
    
    # Create README
    readme_path = val_path / "README_VAL.md"
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # ← FIX: Calculate before f-string
    
    with open(readme_path, 'w') as f:
        f.write(f"""# Validation Dataset

Created: {current_time}

## Split Details
- Validation ratio: {val_ratio*100:.0f}%
- Random seed: {seed}
- Total validation pairs: {val_total}

## Purpose
- Monitor training progress
- Detect overfitting
- Hyperparameter tuning
- Early stopping

## DO NOT USE FOR:
- Training (data leakage)
- Final testing (use test/ folder)
""")
    
    print(f"\n✓ Validation split created successfully!")
    print(f"  Location: {val_path.absolute()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create validation split")
    parser.add_argument('--train-dir', type=str, default='data/train',
                        help='Training directory')
    parser.add_argument('--val-dir', type=str, default='data/val',
                        help='Validation directory')
    parser.add_argument('--val-ratio', type=float, default=0.2,
                        help='Validation ratio (0.2 = 20%%)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    create_validation_split(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        val_ratio=args.val_ratio,
        seed=args.seed
    )
