"""
Data Organization Script for BlueDepth-Crescent
Cleans up messy test folder and organizes 1600 image pairs properly
"""

import shutil
from pathlib import Path
import random
import os

print("=" * 70)
print("BlueDepth-Crescent Data Organization Script")
print("=" * 70)

# ==============================================================================
# STEP 1: VERIFY YOUR ORIGINAL DATA
# ==============================================================================
print("\n[STEP 1] Verifying original data...")

hazy_dir = Path('data/hazy')
clear_dir = Path('data/clear')

# Find all images (jpg, jpeg, png)
hazy_images = sorted(
    list(hazy_dir.glob('*.jpg')) + 
    list(hazy_dir.glob('*.jpeg')) + 
    list(hazy_dir.glob('*.png'))
)
clear_images = sorted(
    list(clear_dir.glob('*.jpg')) + 
    list(clear_dir.glob('*.jpeg')) + 
    list(clear_dir.glob('*.png'))
)

print(f"Found {len(hazy_images)} hazy images in data/hazy/")
print(f"Found {len(clear_images)} clear images in data/clear/")

# Verify counts
if len(hazy_images) != len(clear_images):
    print(f"  WARNING: Image count mismatch!")
    print(f"   Hazy: {len(hazy_images)}, Clear: {len(clear_images)}")
    response = input("Continue anyway? (yes/no): ")
    if response.lower() != 'yes':
        exit(1)

# Verify pairing (check if filenames match)
print("\n[STEP 1.1] Verifying image pairing...")
mismatched = []
for h, c in zip(hazy_images, clear_images):
    if h.name != c.name:
        mismatched.append((h.name, c.name))

if mismatched:
    print(f"  WARNING: {len(mismatched)} filename mismatches found!")
    print("First 5 mismatches:")
    for h, c in mismatched[:5]:
        print(f"  Hazy: {h} ↔ Clear: {c}")
    response = input("Continue anyway? (yes/no): ")
    if response.lower() != 'yes':
        exit(1)
else:
    print(" All filenames match perfectly!")

total_images = len(hazy_images)
print(f"\n Total image pairs: {total_images}")

# ==============================================================================
# STEP 2: CLEAN UP MESSY TEST FOLDER
# ==============================================================================
print("\n[STEP 2] Cleaning up messy test folder...")

test_dir = Path('data/test')
messy_folders = ['clear', 'hazy', 'input', 'output', 'ground_truth']

for folder in messy_folders:
    folder_path = test_dir / folder
    if folder_path.exists():
        try:
            # Count files before deletion
            file_count = len(list(folder_path.glob('*')))
            print(f"  Deleting: {folder_path} ({file_count} files)")
            shutil.rmtree(folder_path)
        except Exception as e:
            print(f"    Error deleting {folder_path}: {e}")

print(" Test folder cleaned!")

# ==============================================================================
# STEP 3: CREATE FRESH FOLDER STRUCTURE
# ==============================================================================
print("\n[STEP 3] Creating fresh folder structure...")

# Create train folders (if they don't exist)
Path('data/train/hazy').mkdir(parents=True, exist_ok=True)
Path('data/train/clear').mkdir(parents=True, exist_ok=True)

# Create fresh test folders
Path('data/test/hazy').mkdir(parents=True, exist_ok=True)
Path('data/test/clear').mkdir(parents=True, exist_ok=True)

# Create other folders
Path('data/raw').mkdir(parents=True, exist_ok=True)
Path('data/enhanced').mkdir(parents=True, exist_ok=True)
Path('data/frames').mkdir(parents=True, exist_ok=True)
Path('data/processed').mkdir(parents=True, exist_ok=True)
Path('data/videos').mkdir(parents=True, exist_ok=True)

print(" Folder structure created!")

# ==============================================================================
# STEP 4: CHECK IF TRAIN FOLDER ALREADY HAS DATA
# ==============================================================================
print("\n[STEP 4] Checking existing training data...")

existing_train_hazy = len(list(Path('data/train/hazy').glob('*')))
existing_train_clear = len(list(Path('data/train/clear').glob('*')))

if existing_train_hazy > 0 or existing_train_clear > 0:
    print(f"Found {existing_train_hazy} hazy and {existing_train_clear} clear images in train/")
    response = input("Clear and recreate train folder? (yes/no): ")
    if response.lower() == 'yes':
        print("  Clearing train folder...")
        shutil.rmtree('data/train/hazy')
        shutil.rmtree('data/train/clear')
        Path('data/train/hazy').mkdir(parents=True, exist_ok=True)
        Path('data/train/clear').mkdir(parents=True, exist_ok=True)
        print(" Train folder cleared!")
    else:
        print("  Keeping existing train data, will only create test set")

# ==============================================================================
# STEP 5: SPLIT DATA (80% TRAIN, 20% TEST)
# ==============================================================================
print("\n[STEP 5] Splitting data (80% train, 20% test)...")

# Shuffle indices for random split
indices = list(range(total_images))
random.seed(42)  # For reproducibility
random.shuffle(indices)

# Calculate split
train_count = int(total_images * 0.8)
test_count = total_images - train_count

train_idx = indices[:train_count]
test_idx = indices[train_count:]

print(f"Training set: {train_count} pairs")
print(f"Testing set: {test_count} pairs")

# ==============================================================================
# STEP 6: COPY TRAINING DATA
# ==============================================================================
print("\n[STEP 6] Copying training data...")

if existing_train_hazy == 0 or existing_train_clear == 0:
    for i, idx in enumerate(train_idx, 1):
        hazy_src = hazy_images[idx]
        clear_src = clear_images[idx]
        
        hazy_dst = Path('data/train/hazy') / hazy_src.name
        clear_dst = Path('data/train/clear') / clear_src.name
        
        shutil.copy(hazy_src, hazy_dst)
        shutil.copy(clear_src, clear_dst)
        
        if i % 100 == 0 or i == len(train_idx):
            print(f"  Copied {i}/{len(train_idx)} training pairs...")
    
    print(f" Training data copied: {train_count} pairs")
else:
    print("⏭  Skipped (keeping existing training data)")

# ==============================================================================
# STEP 7: COPY TESTING DATA
# ==============================================================================
print("\n[STEP 7] Copying testing data...")

for i, idx in enumerate(test_idx, 1):
    hazy_src = hazy_images[idx]
    clear_src = clear_images[idx]
    
    hazy_dst = Path('data/test/hazy') / hazy_src.name
    clear_dst = Path('data/test/clear') / clear_src.name
    
    shutil.copy(hazy_src, hazy_dst)
    shutil.copy(clear_src, clear_dst)
    
    if i % 100 == 0 or i == len(test_idx):
        print(f"  Copied {i}/{len(test_idx)} testing pairs...")

print(f" Testing data copied: {test_count} pairs")

# ==============================================================================
# STEP 8: BACKUP ORIGINALS TO RAW FOLDER (OPTIONAL)
# ==============================================================================
print("\n[STEP 8] Backing up originals to data/raw/...")

raw_hazy = Path('data/raw/hazy')
raw_clear = Path('data/raw/clear')
raw_hazy.mkdir(parents=True, exist_ok=True)
raw_clear.mkdir(parents=True, exist_ok=True)

response = input("Backup all originals to data/raw/? (yes/no): ")
if response.lower() == 'yes':
    print("  Backing up hazy images...")
    for img in hazy_images:
        shutil.copy(img, raw_hazy / img.name)
    
    print("  Backing up clear images...")
    for img in clear_images:
        shutil.copy(img, raw_clear / img.name)
    
    print(f" Backup complete: {total_images} pairs saved to data/raw/")
else:
    print("⏭  Backup skipped")

# ==============================================================================
# STEP 9: VERIFY FINAL STRUCTURE
# ==============================================================================
print("\n[STEP 9] Verifying final structure...")

final_counts = {
    'train/hazy': len(list(Path('data/train/hazy').glob('*'))),
    'train/clear': len(list(Path('data/train/clear').glob('*'))),
    'test/hazy': len(list(Path('data/test/hazy').glob('*'))),
    'test/clear': len(list(Path('data/test/clear').glob('*'))),
}

print("\nFinal counts:")
for folder, count in final_counts.items():
    print(f"  data/{folder}: {count} images")

# Verify pairing in train and test
train_hazy_files = sorted([f.name for f in Path('data/train/hazy').glob('*')])
train_clear_files = sorted([f.name for f in Path('data/train/clear').glob('*')])
test_hazy_files = sorted([f.name for f in Path('data/test/hazy').glob('*')])
test_clear_files = sorted([f.name for f in Path('data/test/clear').glob('*')])

if train_hazy_files == train_clear_files:
    print(" Training set: All filenames match!")
else:
    print("  Training set: Filename mismatch detected!")

if test_hazy_files == test_clear_files:
    print(" Testing set: All filenames match!")
else:
    print("  Testing set: Filename mismatch detected!")

# ==============================================================================
# STEP 10: DONE!
# ==============================================================================
print("\n" + "=" * 70)
print(" DATA ORGANIZATION COMPLETE!")
print("=" * 70)
print("\nYour data structure is now:")
print("""
data/
 train/
    hazy/     ({} images)
    clear/    ({} images)
 test/
    hazy/     ({} images)
    clear/    ({} images)
 raw/          (originals backup)
 enhanced/     (model outputs)
 frames/       (video frames)
 videos/       (input videos)
""".format(
    final_counts['train/hazy'],
    final_counts['train/clear'],
    final_counts['test/hazy'],
    final_counts['test/clear']
))

print("\n You can now start training:")
print("   python -m training.train_unet --model attention --data data/train --batch-size 4")
