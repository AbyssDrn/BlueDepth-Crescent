#!/usr/bin/env python3
"""
BlueDepth-Crescent Dataset Setup Script
Downloads, organizes, and validates UIEB dataset
"""

import os
import sys
import shutil
import zipfile
import requests
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm
import random

# ANSI colors
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text: str):
    """Print colored header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text:^70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}\n")

def print_success(text: str):
    print(f"{Colors.GREEN} {text}{Colors.END}")

def print_error(text: str):
    print(f"{Colors.RED} {text}{Colors.END}")

def print_warning(text: str):
    print(f"{Colors.YELLOW} {text}{Colors.END}")

def create_directory_structure():
    """Create required directory structure"""
    print_header("Creating Directory Structure")
    
    dirs = [
        'data/raw',
        'data/train/hazy',
        'data/train/clear',
        'data/test/hazy',
        'data/test/clear',
        'data/videos',
        'data/frames',
        'data/enhanced',
        'data/processed',
        'checkpoints',
        'logs',
        'results'
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print_success(f"Created: {dir_path}")

def download_file(url: str, destination: Path) -> bool:
    """Download file with progress bar"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
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
        
        return True
    except Exception as e:
        print_error(f"Download failed: {e}")
        return False

def organize_uieb_dataset(raw_dir: Path, train_dir: Path, test_dir: Path, split_ratio: float = 0.8):
    """Organize UIEB dataset into train/test split"""
    print_header("Organizing UIEB Dataset")
    
    # Expected UIEB structure
    hazy_dir = raw_dir / "raw-890"
    clear_dir = raw_dir / "reference-890"
    
    if not hazy_dir.exists() or not clear_dir.exists():
        print_error("UIEB dataset structure not found!")
        print_warning("Expected structure:")
        print_warning("  data/raw/raw-890/ (hazy images)")
        print_warning("  data/raw/reference-890/ (clear images)")
        return False
    
    # Get all image pairs
    hazy_images = sorted(list(hazy_dir.glob("*.png")))
    clear_images = sorted(list(clear_dir.glob("*.png")))
    
    if len(hazy_images) != len(clear_images):
        print_error(f"Mismatch: {len(hazy_images)} hazy vs {len(clear_images)} clear")
        return False
    
    print_success(f"Found {len(hazy_images)} image pairs")
    
    # Shuffle and split
    pairs = list(zip(hazy_images, clear_images))
    random.shuffle(pairs)
    
    split_idx = int(len(pairs) * split_ratio)
    train_pairs = pairs[:split_idx]
    test_pairs = pairs[split_idx:]
    
    print_success(f"Train set: {len(train_pairs)} pairs")
    print_success(f"Test set: {len(test_pairs)} pairs")
    
    # Copy training data
    print("\nCopying training data...")
    for hazy, clear in tqdm(train_pairs, desc="Train"):
        shutil.copy(hazy, train_dir / "hazy" / hazy.name)
        shutil.copy(clear, train_dir / "clear" / clear.name)
    
    # Copy test data
    print("Copying test data...")
    for hazy, clear in tqdm(test_pairs, desc="Test"):
        shutil.copy(hazy, test_dir / "hazy" / hazy.name)
        shutil.copy(clear, test_dir / "clear" / clear.name)
    
    print_success("Dataset organized successfully!")
    return True

def validate_dataset(train_dir: Path, test_dir: Path) -> bool:
    """Validate organized dataset"""
    print_header("Validating Dataset")
    
    checks = [
        (train_dir / "hazy", "Training hazy images"),
        (train_dir / "clear", "Training clear images"),
        (test_dir / "hazy", "Test hazy images"),
        (test_dir / "clear", "Test clear images")
    ]
    
    all_valid = True
    
    for path, description in checks:
        images = list(path.glob("*.png")) + list(path.glob("*.jpg"))
        count = len(images)
        
        if count > 0:
            print_success(f"{description:30} {count:4d} images")
        else:
            print_error(f"{description:30} EMPTY!")
            all_valid = False
    
    # Check for matching pairs
    train_hazy = set([p.stem for p in (train_dir / "hazy").glob("*")])
    train_clear = set([p.stem for p in (train_dir / "clear").glob("*")])
    test_hazy = set([p.stem for p in (test_dir / "hazy").glob("*")])
    test_clear = set([p.stem for p in (test_dir / "clear").glob("*")])
    
    if train_hazy == train_clear:
        print_success("Training pairs match")
    else:
        print_error(f"Training mismatch: {len(train_hazy)} hazy vs {len(train_clear)} clear")
        all_valid = False
    
    if test_hazy == test_clear:
        print_success("Test pairs match")
    else:
        print_error(f"Test mismatch: {len(test_hazy)} hazy vs {len(test_clear)} clear")
        all_valid = False
    
    return all_valid

def generate_dataset_info(train_dir: Path, test_dir: Path):
    """Generate dataset information file"""
    print_header("Generating Dataset Info")
    
    train_count = len(list((train_dir / "hazy").glob("*")))
    test_count = len(list((test_dir / "hazy").glob("*")))
    total_count = train_count + test_count
    
    info = f"""
BlueDepth-Crescent Dataset Information
=====================================

Dataset: UIEB (Underwater Image Enhancement Benchmark)
Organization Date: {Path(__file__).stat().st_mtime}

Structure:
---------
Training Set:   {train_count} image pairs
Test Set:       {test_count} image pairs
Total:          {total_count} image pairs

Split Ratio: {train_count/total_count:.1%} train / {test_count/total_count:.1%} test

Directory Structure:
-------------------
data/
 train/
    hazy/    ({train_count} images)
    clear/   ({train_count} images)
 test/
     hazy/    ({test_count} images)
     clear/   ({test_count} images)

Ready for training!
"""
    
    with open('data/DATASET_INFO.txt', 'w') as f:
        f.write(info)
    
    print(info)
    print_success("Dataset info saved to data/DATASET_INFO.txt")

def main():
    """Main setup function"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}")
    print("")
    print("       BlueDepth-Crescent Dataset Setup Script v1.0            ")
    print("")
    print(f"{Colors.END}\n")
    
    # Step 1: Create directories
    create_directory_structure()
    
    # Step 2: Check for raw data
    raw_dir = Path("data/raw")
    train_dir = Path("data/train")
    test_dir = Path("data/test")
    
    if not (raw_dir / "raw-890").exists():
        print_warning("\nUIEB dataset not found in data/raw/")
        print_warning("Please download UIEB dataset and extract to data/raw/")
        print_warning("\nExpected structure:")
        print_warning("  data/raw/raw-890/        (890 hazy images)")
        print_warning("  data/raw/reference-890/  (890 clear images)")
        print_warning("\nDownload from: https://li-chongyi.github.io/proj_benchmark.html")
        return 1
    
    # Step 3: Organize dataset
    if not organize_uieb_dataset(raw_dir, train_dir, test_dir, split_ratio=0.8):
        return 1
    
    # Step 4: Validate
    if not validate_dataset(train_dir, test_dir):
        print_error("Dataset validation failed!")
        return 1
    
    # Step 5: Generate info
    generate_dataset_info(train_dir, test_dir)
    
    print_header("SETUP COMPLETE")
    print_success("Dataset ready for training!")
    print(f"\n{Colors.BOLD}Next steps:{Colors.END}")
    print(f"  1. Validate installation: python scripts/validate_installation.py")
    print(f"  2. Start training: python -m training.train_unet --config configs/rtx4050.yaml\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
