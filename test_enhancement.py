#!/usr/bin/env python3
"""
Quick Test Script for BlueDepth-Crescent
Tests your trained model on sample images
"""
import sys
from pathlib import Path
from inference.enhancer import ImageEnhancer
from PIL import Image

def find_test_images():
    """Find available images to test"""
    search_dirs = [
        Path("data/hazy"),
        Path("data/raw"),
        Path("data/test"),
        Path("data")
    ]
    
    images = []
    for dir_path in search_dirs:
        if dir_path.exists():
            images.extend(list(dir_path.glob("*.jpg")))
            images.extend(list(dir_path.glob("*.png")))
            images.extend(list(dir_path.glob("*.jpeg")))
    
    return images

def find_checkpoint():
    """Find best checkpoint"""
    checkpoint_dir = Path("checkpoints")
    
    # Look for best checkpoint
    best = list(checkpoint_dir.glob("*_best.pth"))
    if best:
        return str(best[0])
    
    # Or latest epoch
    epochs = sorted(checkpoint_dir.glob("*_epoch_*.pth"))
    if epochs:
        return str(epochs[-1])
    
    return None

def main():
    print("=" * 70)
    print(" BlueDepth-Crescent - Quick Test")
    print("=" * 70)
    
    # Find checkpoint
    print("\n Searching for model checkpoint...")
    checkpoint_path = find_checkpoint()
    
    if not checkpoint_path:
        print(" No checkpoint found! Train a model first:")
        print("   python main.py train")
        sys.exit(1)
    
    print(f" Found checkpoint: {Path(checkpoint_path).name}")
    
    # Find test images
    print("\n Searching for test images...")
    images = find_test_images()
    
    if not images:
        print(" No images found!")
        print("\nPlease add images to one of these folders:")
        print("  - data/hazy/")
        print("  - data/raw/")
        print("  - data/test/")
        sys.exit(1)
    
    print(f" Found {len(images)} images")
    
    # Show available images
    print("\n Available Images:")
    for i, img in enumerate(images[:10], 1):  # Show first 10
        print(f"   {i}. {img.parent.name}/{img.name}")
    
    if len(images) > 10:
        print(f"   ... and {len(images) - 10} more")
    
    # Select image
    print("\n" + "=" * 70)
    if len(images) == 1:
        selected = images[0]
        print(f"Processing: {selected.name}")
    else:
        try:
            choice = input(f"\nSelect image (1-{min(len(images), 10)}) or press Enter for first: ").strip()
            if not choice:
                selected = images[0]
            else:
                idx = int(choice) - 1
                selected = images[idx]
        except (ValueError, IndexError):
            selected = images[0]
    
    print(f"\n Processing: {selected}")
    print("=" * 70)
    
    # Load model
    print("\n⏳ Loading model...")
    try:
        enhancer = ImageEnhancer(checkpoint_path)
        print(" Model loaded successfully!")
    except Exception as e:
        print(f" Error loading model: {e}")
        sys.exit(1)
    
    # Enhance image
    print("\n⏳ Enhancing image...")
    try:
        enhanced = enhancer.enhance_image(str(selected))
        print(" Enhancement complete!")
    except Exception as e:
        print(f" Error during enhancement: {e}")
        sys.exit(1)
    
    # Save result
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    output_path = results_dir / f"enhanced_{selected.name}"
    enhanced.save(output_path)
    
    print(f"\n Saved to: {output_path}")
    
    # Show before/after info
    original = Image.open(selected)
    
    print("\n" + "=" * 70)
    print(" Results:")
    print("=" * 70)
    print(f"Original:  {original.size[0]} x {original.size[1]} px")
    print(f"Enhanced:  {enhanced.size[0]} x {enhanced.size[1]} px")
    print(f"Location:  {output_path.absolute()}")
    
    print("\n" + "=" * 70)
    print(" Test Complete!")
    print("=" * 70)
    print(f"\nTo enhance more images:")
    print(f"  python main.py enhance --input \"path/to/image.jpg\" --output \"results/output.jpg\"")
    print(f"\nOr batch process:")
    print(f"  python main.py batch --input_dir \"data/raw\" --output_dir \"results\"")
    print()

if __name__ == "__main__":
    main()