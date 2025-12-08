"""
Image Processing Utilities for BlueDepth-Crescent
Handles image loading, saving, transformations, and comparisons
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from pathlib import Path
from typing import Union, Tuple, Optional, List
import torch

def load_image(
    path: Union[str, Path],
    size: Optional[Tuple[int, int]] = None,
    mode: str = 'RGB'
) -> np.ndarray:
    """
    Load image from path
    
    Args:
        path: Image file path
        size: Optional resize (width, height)
        mode: Color mode ('RGB', 'L', 'RGBA')
    
    Returns:
        Numpy array [H, W, C]
    """
    img = Image.open(path).convert(mode)
    
    if size:
        img = img.resize(size, Image.LANCZOS)
    
    return np.array(img)

def save_image(
    img: Union[np.ndarray, torch.Tensor, Image.Image],
    path: Union[str, Path],
    quality: int = 95
) -> None:
    """
    Save image to path
    
    Args:
        img: Image (numpy array, torch tensor, or PIL Image)
        path: Output path
        quality: JPEG quality (1-100)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to PIL Image
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
    
    if isinstance(img, np.ndarray):
        # Normalize to [0, 255]
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
        
        # Handle channel order
        if img.ndim == 3 and img.shape[0] == 3:  # CHW format
            img = np.transpose(img, (1, 2, 0))
        
        img = Image.fromarray(img)
    
    # Save
    if path.suffix.lower() in ['.jpg', '.jpeg']:
        img.save(path, quality=quality, optimize=True)
    else:
        img.save(path)

def resize_image(
    img: np.ndarray,
    size: Tuple[int, int],
    interpolation: int = cv2.INTER_LANCZOS4
) -> np.ndarray:
    """
    Resize image using OpenCV
    
    Args:
        img: Input image
        size: Target (width, height)
        interpolation: Interpolation method
    
    Returns:
        Resized image
    """
    return cv2.resize(img, size, interpolation=interpolation)

def normalize(img: np.ndarray, mean: Optional[List[float]] = None, std: Optional[List[float]] = None) -> np.ndarray:
    """
    Normalize image to [0, 1] or using mean/std
    
    Args:
        img: Input image [0, 255]
        mean: Optional mean values
        std: Optional std values
    
    Returns:
        Normalized image
    """
    img = img.astype(np.float32) / 255.0
    
    if mean and std:
        mean = np.array(mean).reshape(1, 1, 3)
        std = np.array(std).reshape(1, 1, 3)
        img = (img - mean) / std
    
    return img

def denormalize(
    img: np.ndarray,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None
) -> np.ndarray:
    """
    Denormalize image back to [0, 255]
    
    Args:
        img: Normalized image
        mean: Optional mean values used in normalization
        std: Optional std values used in normalization
    
    Returns:
        Image in [0, 255] range
    """
    if mean and std:
        mean = np.array(mean).reshape(1, 1, 3)
        std = np.array(std).reshape(1, 1, 3)
        img = (img * std) + mean
    
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img

def create_comparison(
    original: Union[np.ndarray, Image.Image],
    enhanced: Union[np.ndarray, Image.Image],
    labels: Tuple[str, str] = ("Original", "Enhanced"),
    font_size: int = 30
) -> Image.Image:
    """
    Create side-by-side comparison image with labels
    
    Args:
        original: Original image
        enhanced: Enhanced image
        labels: Text labels for images
        font_size: Font size for labels
    
    Returns:
        Comparison image
    """
    # Convert to PIL if needed
    if isinstance(original, np.ndarray):
        if original.max() <= 1.0:
            original = (original * 255).astype(np.uint8)
        original = Image.fromarray(original)
    
    if isinstance(enhanced, np.ndarray):
        if enhanced.max() <= 1.0:
            enhanced = (enhanced * 255).astype(np.uint8)
        enhanced = Image.fromarray(enhanced)
    
    # Resize to match
    width, height = original.size
    enhanced = enhanced.resize((width, height), Image.LANCZOS)
    
    # Create comparison canvas
    comparison = Image.new('RGB', (width * 2, height + font_size + 10))
    
    # Paste images
    comparison.paste(original, (0, font_size + 10))
    comparison.paste(enhanced, (width, font_size + 10))
    
    # Add labels
    draw = ImageDraw.Draw(comparison)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    draw.text((width // 2 - 50, 5), labels[0], fill='white', font=font)
    draw.text((width + width // 2 - 50, 5), labels[1], fill='white', font=font)
    
    return comparison

def calculate_image_stats(img: np.ndarray) -> dict:
    """
    Calculate image statistics
    
    Args:
        img: Input image
    
    Returns:
        Dictionary with statistics
    """
    if img.max() > 1.0:
        img = img / 255.0
    
    return {
        'mean': float(np.mean(img)),
        'std': float(np.std(img)),
        'min': float(np.min(img)),
        'max': float(np.max(img)),
        'median': float(np.median(img)),
        'brightness': float(np.mean(img) * 100)  # Percentage
    }

def adjust_brightness(img: np.ndarray, factor: float) -> np.ndarray:
    """Adjust image brightness"""
    return np.clip(img * factor, 0, 255).astype(np.uint8)

def adjust_contrast(img: np.ndarray, factor: float) -> np.ndarray:
    """Adjust image contrast"""
    mean = np.mean(img)
    return np.clip((img - mean) * factor + mean, 0, 255).astype(np.uint8)
