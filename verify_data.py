#!/usr/bin/env python3
"""
Evaluate Test Set Performance
Compare enhanced outputs with ground truth (if available)
"""
from pathlib import Path
from PIL import Image
import numpy as np
from utils.metrics import calculate_psnr, calculate_ssim, calculate_mae

def evaluate_test_set():
    print("=" * 80)
    print(" EVALUATING TEST SET")
    print("=" * 80)
    print()
    
    input_dir = Path("data/test/input")
    output_dir = Path("data/test/output")
    ground_truth_dir = Path("data/test/ground_truth")
    
    # Check if we have ground truth
    has_ground_truth = ground_truth_dir.exists() and len(list(ground_truth_dir.glob("*.jpg"))) > 0
    
    if not has_ground_truth:
        print("ℹ  No ground truth found - Visual comparison only")
        print()
        print(" Enhanced images saved to: data/test/output/")
        print(" Compare visually:")
        print("   - Original: data/test/input/")
        print("   - Enhanced: data/test/output/")
        return
    
    print(" Ground truth found - Computing metrics...")
    print()
    
    # Get all images
    input_images = sorted(list(input_dir.glob("*.jpg")))
    output_images = sorted(list(output_dir.glob("*.jpg")))
    gt_images = sorted(list(ground_truth_dir.glob("*.jpg")))
    
    if len(output_images) == 0:
        print(" No enhanced images found in data/test/output/")
        print("   Run enhancement first: python main.py batch --input_dir data/test/input --output_dir data/test/output")
        return
    
    print(f" Found {len(output_images)} enhanced images to evaluate")
    print()
    
    # Calculate metrics
    psnr_scores = []
    ssim_scores = []
    mae_scores = []
    
    for output_path in output_images:
        gt_path = ground_truth_dir / output_path.name
        
        if not gt_path.exists():
            print(f"  Skipping {output_path.name} - no ground truth match")
            continue
        
        # Load images
        output_img = np.array(Image.open(output_path))
        gt_img = np.array(Image.open(gt_path))
        
        # Calculate metrics
        psnr = calculate_psnr(output_img, gt_img)
        ssim = calculate_ssim(output_img, gt_img)
        mae = calculate_mae(output_img, gt_img)
        
        psnr_scores.append(psnr)
        ssim_scores.append(ssim)
        mae_scores.append(mae)
    
    # Print results
    print("=" * 80)
    print(" RESULTS")
    print("=" * 80)
    print()
    print(f"Images evaluated: {len(psnr_scores)}")
    print()
    print(f" Average PSNR: {np.mean(psnr_scores):.2f} dB")
    print(f"   (Higher is better, >30 is good)")
    print()
    print(f" Average SSIM: {np.mean(ssim_scores):.4f}")
    print(f"   (Range: 0-1, >0.8 is good)")
    print()
    print(f" Average MAE:  {np.mean(mae_scores):.2f}")
    print(f"   (Lower is better)")
    print()
    print("=" * 80)
    
    # Quality assessment
    avg_psnr = np.mean(psnr_scores)
    avg_ssim = np.mean(ssim_scores)
    
    print(" QUALITY ASSESSMENT:")
    if avg_psnr > 30 and avg_ssim > 0.8:
        print("    EXCELLENT - Model performing very well!")
    elif avg_psnr > 25 and avg_ssim > 0.7:
        print("    GOOD - Model performing well")
    elif avg_psnr > 20 and avg_ssim > 0.6:
        print("     FAIR - Model needs improvement")
    else:
        print("    POOR - Consider retraining with more data")
    
    print("=" * 80)

if __name__ == "__main__":
    try:
        from utils.metrics import calculate_psnr, calculate_ssim, calculate_mae
        evaluate_test_set()
    except ImportError:
        print(" Error: Metrics module not found")
        print("   Make sure utils/metrics.py exists")