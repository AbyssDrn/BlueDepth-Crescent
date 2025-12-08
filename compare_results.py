#!/usr/bin/env python3
"""
Compare Original vs Enhanced Images
Creates side-by-side comparison visualizations
"""
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import numpy as np

def compare_images(original_path, enhanced_path, save_path=None):
    """Create side-by-side comparison"""
    original = Image.open(original_path)
    enhanced = Image.open(enhanced_path)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Original
    axes[0].imshow(original)
    axes[0].set_title('Original (Degraded)', fontsize=16, fontweight='bold')
    axes[0].axis('off')
    
    # Enhanced
    axes[1].imshow(enhanced)
    axes[1].set_title('Enhanced (BlueDepth-Crescent)', fontsize=16, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f" Saved comparison to: {save_path}")
    else:
        plt.show()
    
    plt.close()

def create_comparison_grid(image_pairs, grid_size=(2, 3), save_path=None):
    """Create a grid of comparisons"""
    rows, cols = grid_size
    total = rows * cols
    
    if len(image_pairs) < total:
        total = len(image_pairs)
        rows = (total + cols - 1) // cols
    
    fig = plt.figure(figsize=(cols * 6, rows * 4))
    
    for idx, (orig, enh) in enumerate(image_pairs[:total]):
        # Original
        ax1 = plt.subplot(rows, cols * 2, idx * 2 + 1)
        img1 = Image.open(orig)
        ax1.imshow(img1)
        ax1.set_title(f'Original #{idx+1}', fontsize=10)
        ax1.axis('off')
        
        # Enhanced
        ax2 = plt.subplot(rows, cols * 2, idx * 2 + 2)
        img2 = Image.open(enh)
        ax2.imshow(img2)
        ax2.set_title(f'Enhanced #{idx+1}', fontsize=10)
        ax2.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f" Saved grid comparison to: {save_path}")
    else:
        plt.show()
    
    plt.close()

def calculate_metrics(original_path, enhanced_path):
    """Calculate quality metrics"""
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim
    
    orig = np.array(Image.open(original_path))
    enh = np.array(Image.open(enhanced_path))
    
    # Ensure same size
    if orig.shape != enh.shape:
        enh = np.array(Image.open(enhanced_path).resize(
            (orig.shape[1], orig.shape[0])
        ))
    
    # Calculate metrics
    psnr_val = psnr(orig, enh, data_range=255)
    ssim_val = ssim(orig, enh, multichannel=True, channel_axis=2)
    
    return {
        'psnr': psnr_val,
        'ssim': ssim_val
    }

def main():
    print("=" * 70)
    print(" BlueDepth-Crescent - Image Comparison Tool")
    print("=" * 70)
    
    # Find image pairs
    original_dir = Path("data/hazy")
    enhanced_dir = Path("results")
    
    if not enhanced_dir.exists():
        print("\n No enhanced images found!")
        print("Run enhancement first:")
        print("  python test_enhancement.py")
        return
    
    # Find matching pairs
    enhanced_images = list(enhanced_dir.glob("enhanced_*.jpg"))
    
    if not enhanced_images:
        print("\n No enhanced images found in results/")
        return
    
    print(f"\n Found {len(enhanced_images)} enhanced images")
    
    # Create comparison for first few images
    print("\n Creating comparisons...")
    
    comparison_dir = Path("results/comparisons")
    comparison_dir.mkdir(exist_ok=True)
    
    pairs = []
    
    for enh_path in enhanced_images[:10]:  # First 10
        # Find original
        orig_name = enh_path.name.replace("enhanced_", "")
        orig_path = original_dir / orig_name
        
        if orig_path.exists():
            pairs.append((orig_path, enh_path))
            
            # Create individual comparison
            compare_path = comparison_dir / f"comparison_{orig_name}"
            compare_images(orig_path, enh_path, compare_path)
            
            # Calculate metrics
            try:
                metrics = calculate_metrics(orig_path, enh_path)
                print(f"  {orig_name}: PSNR={metrics['psnr']:.2f}dB, SSIM={metrics['ssim']:.4f}")
            except:
                print(f"  {orig_name}: Comparison created")
    
    # Create grid comparison
    if len(pairs) >= 6:
        print("\n Creating grid comparison...")
        grid_path = comparison_dir / "comparison_grid.png"
        create_comparison_grid(pairs[:6], grid_size=(2, 3), save_path=grid_path)
    
    print("\n" + "=" * 70)
    print(" Comparison Complete!")
    print("=" * 70)
    print(f"\n Comparisons saved to: {comparison_dir.absolute()}")
    print("\nNext steps:")
    print("1. Check results/comparisons/ folder")
    print("2. Batch process all images: python main.py batch --input_dir data/hazy --output_dir results/batch")
    print("3. Launch dashboard: python main.py dashboard")
    print()

if __name__ == "__main__":
    main()