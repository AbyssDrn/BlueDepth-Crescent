"""
Image Quality Metrics for BlueDepth-Crescent
PSNR, SSIM, MAE, MSE, and comprehensive evaluation
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Union, Dict, Tuple
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import cv2

def calculate_psnr(
    img1: Union[np.ndarray, torch.Tensor],
    img2: Union[np.ndarray, torch.Tensor],
    data_range: float = 255.0
) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR)
    
    Args:
        img1: First image
        img2: Second image
        data_range: Data range (255 for uint8, 1.0 for float)
    
    Returns:
        PSNR value in dB (higher is better)
    """
    if isinstance(img1, torch.Tensor):
        img1 = img1.cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.cpu().numpy()
    
    # Ensure same shape
    if img1.shape != img2.shape:
        raise ValueError(f"Shape mismatch: {img1.shape} vs {img2.shape}")
    
    # Handle channel-first format
    if img1.ndim == 3 and img1.shape[0] == 3:
        img1 = np.transpose(img1, (1, 2, 0))
    if img2.ndim == 3 and img2.shape[0] == 3:
        img2 = np.transpose(img2, (1, 2, 0))
    
    return peak_signal_noise_ratio(img1, img2, data_range=data_range)

def calculate_ssim(
    img1: Union[np.ndarray, torch.Tensor],
    img2: Union[np.ndarray, torch.Tensor],
    data_range: float = 255.0
) -> float:
    """
    Calculate Structural Similarity Index (SSIM)
    
    Args:
        img1: First image
        img2: Second image
        data_range: Data range
    
    Returns:
        SSIM value [0, 1] (higher is better)
    """
    if isinstance(img1, torch.Tensor):
        img1 = img1.cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.cpu().numpy()
    
    # Handle channel-first format
    if img1.ndim == 3 and img1.shape[0] == 3:
        img1 = np.transpose(img1, (1, 2, 0))
    if img2.ndim == 3 and img2.shape[0] == 3:
        img2 = np.transpose(img2, (1, 2, 0))
    
    # For multi-channel images
    if img1.ndim == 3:
        return structural_similarity(
            img1, img2,
            data_range=data_range,
            channel_axis=2
        )
    else:
        return structural_similarity(img1, img2, data_range=data_range)

def calculate_mae(
    img1: Union[np.ndarray, torch.Tensor],
    img2: Union[np.ndarray, torch.Tensor]
) -> float:
    """
    Calculate Mean Absolute Error (MAE)
    
    Args:
        img1: First image
        img2: Second image
    
    Returns:
        MAE value (lower is better)
    """
    if isinstance(img1, torch.Tensor):
        img1 = img1.cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.cpu().numpy()
    
    return float(np.mean(np.abs(img1.astype(float) - img2.astype(float))))

def calculate_mse(
    img1: Union[np.ndarray, torch.Tensor],
    img2: Union[np.ndarray, torch.Tensor]
) -> float:
    """
    Calculate Mean Squared Error (MSE)
    
    Args:
        img1: First image
        img2: Second image
    
    Returns:
        MSE value (lower is better)
    """
    if isinstance(img1, torch.Tensor):
        img1 = img1.cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.cpu().numpy()
    
    return float(np.mean((img1.astype(float) - img2.astype(float)) ** 2))

def calculate_uiqm(img: np.ndarray) -> float:
    """
    Calculate Underwater Image Quality Measure (UIQM)
    Specialized metric for underwater images
    
    Args:
        img: Underwater image (RGB, 0-255)
    
    Returns:
        UIQM score (higher is better)
    """
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    
    # Colorfulness (UICM)
    rg = img[:, :, 0].astype(float) - img[:, :, 1].astype(float)
    yb = (img[:, :, 0].astype(float) + img[:, :, 1].astype(float)) / 2 - img[:, :, 2].astype(float)
    uicm = np.sqrt(np.mean(rg**2) + np.mean(yb**2))
    
    # Sharpness (UISM)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    uism = np.sqrt(np.mean(sobel_x**2 + sobel_y**2))
    
    # Contrast (UIConM)
    uiconm = np.std(lab[:, :, 0])
    
    # Weighted combination
    uiqm = 0.0282 * uicm + 0.2953 * uism + 3.5753 * uiconm
    
    return float(uiqm)

class MetricsCalculator:
    """Comprehensive metrics calculator"""
    
    @staticmethod
    def compute_all(
        pred: Union[np.ndarray, torch.Tensor],
        target: Union[np.ndarray, torch.Tensor],
        data_range: float = 255.0
    ) -> Dict[str, float]:
        """
        Compute all available metrics
        
        Args:
            pred: Predicted/enhanced image
            target: Ground truth image
            data_range: Data range of images
        
        Returns:
            Dictionary with all metrics
        """
        metrics = {
            'psnr': calculate_psnr(pred, target, data_range),
            'ssim': calculate_ssim(pred, target, data_range),
            'mae': calculate_mae(pred, target),
            'mse': calculate_mse(pred, target)
        }
        
        # Add UIQM if pred is numpy array
        if isinstance(pred, np.ndarray):
            if pred.max() <= 1.0:
                pred_uint8 = (pred * 255).astype(np.uint8)
            else:
                pred_uint8 = pred.astype(np.uint8)
            
            if pred_uint8.ndim == 3 and pred_uint8.shape[0] == 3:
                pred_uint8 = np.transpose(pred_uint8, (1, 2, 0))
            
            try:
                metrics['uiqm'] = calculate_uiqm(pred_uint8)
            except:
                pass
        
        return metrics
    
    @staticmethod
    def print_metrics(metrics: Dict[str, float]):
        """Pretty print metrics"""
        print("\n" + "="*50)
        print("Image Quality Metrics")
        print("="*50)
        for metric, value in metrics.items():
            print(f"{metric.upper():10s}: {value:8.4f}")
        print("="*50 + "\n")

def evaluate_enhancement(
    original: np.ndarray,
    enhanced: np.ndarray,
    ground_truth: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Comprehensive evaluation of image enhancement
    
    Args:
        original: Original degraded image
        enhanced: Enhanced image
        ground_truth: Optional ground truth for reference metrics
    
    Returns:
        Dictionary with evaluation results
    """
    results = {}
    
    # Enhancement metrics (no reference needed)
    results['enhanced_uiqm'] = calculate_uiqm(enhanced)
    results['original_uiqm'] = calculate_uiqm(original)
    results['uiqm_improvement'] = results['enhanced_uiqm'] - results['original_uiqm']
    
    # Reference metrics (if ground truth available)
    if ground_truth is not None:
        ref_metrics = MetricsCalculator.compute_all(enhanced, ground_truth)
        results.update({f'ref_{k}': v for k, v in ref_metrics.items()})
    
    return results
