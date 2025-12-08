"""
Visualization Utilities for BlueDepth-Crescent
Plot training curves, comparisons, and create visualizations
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union, List, Tuple, Optional, Dict
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

def plot_comparison(
    original: np.ndarray,
    enhanced: np.ndarray,
    title: str = "Image Enhancement Comparison",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot original vs enhanced image side-by-side
    
    Args:
        original: Original image
        enhanced: Enhanced image
        title: Plot title
        save_path: Optional path to save figure
    
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Handle normalization
    if original.max() <= 1.0:
        original = (original * 255).astype(np.uint8)
    if enhanced.max() <= 1.0:
        enhanced = (enhanced * 255).astype(np.uint8)
    
    # Handle channel-first format
    if original.ndim == 3 and original.shape[0] == 3:
        original = np.transpose(original, (1, 2, 0))
    if enhanced.ndim == 3 and enhanced.shape[0] == 3:
        enhanced = np.transpose(enhanced, (1, 2, 0))
    
    # Plot
    axes[0].imshow(original)
    axes[0].set_title("Original (Hazy)", fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(enhanced)
    axes[1].set_title("Enhanced", fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig

def plot_training_curves(
    losses: List[float],
    title: str = "Training Loss",
    xlabel: str = "Epoch",
    ylabel: str = "Loss",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot training loss curves
    
    Args:
        losses: List of loss values
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: Optional save path
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    epochs = range(1, len(losses) + 1)
    ax.plot(epochs, losses, 'b-', linewidth=2, label='Training Loss')
    
    # Add trend line
    z = np.polyfit(epochs, losses, 2)
    p = np.poly1d(z)
    ax.plot(epochs, p(epochs), 'r--', linewidth=1, alpha=0.7, label='Trend')
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig

def plot_loss_curves(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot training and validation loss curves
    
    Args:
        train_losses: Training losses
        val_losses: Validation losses
        save_path: Optional save path
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, 'b-', linewidth=2, label='Training Loss')
    
    if val_losses:
        ax.plot(epochs, val_losses, 'r-', linewidth=2, label='Validation Loss')
        
        # Mark best epoch
        best_epoch = np.argmin(val_losses) + 1
        best_loss = min(val_losses)
        ax.scatter([best_epoch], [best_loss], color='green', s=100, zorder=5, label=f'Best (Epoch {best_epoch})')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Progress', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig

def plot_metrics(
    metrics: Dict[str, List[float]],
    title: str = "Training Metrics",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot multiple metrics over time
    
    Args:
        metrics: Dictionary of metric_name: [values]
        title: Plot title
        save_path: Optional save path
    
    Returns:
        Matplotlib figure
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 5))
    
    if n_metrics == 1:
        axes = [axes]
    
    for ax, (metric_name, values) in zip(axes, metrics.items()):
        epochs = range(1, len(values) + 1)
        ax.plot(epochs, values, linewidth=2)
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel(metric_name.upper(), fontsize=10)
        ax.set_title(metric_name.upper(), fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig

def save_comparison_grid(
    images: List[Tuple[np.ndarray, str]],
    output_path: Union[str, Path],
    cols: int = 3
) -> None:
    """
    Save a grid of images with labels
    
    Args:
        images: List of (image, label) tuples
        output_path: Output file path
        cols: Number of columns
    """
    n_images = len(images)
    rows = (n_images + cols - 1) // cols
    
    fig = plt.figure(figsize=(5*cols, 5*rows))
    gs = gridspec.GridSpec(rows, cols, hspace=0.3, wspace=0.2)
    
    for idx, (img, label) in enumerate(images):
        ax = fig.add_subplot(gs[idx])
        
        # Handle normalization
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        
        # Handle channel-first
        if img.ndim == 3 and img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        
        ax.imshow(img)
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.axis('off')
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_histogram_comparison(
    original: np.ndarray,
    enhanced: np.ndarray,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot histogram comparison of original vs enhanced
    
    Args:
        original: Original image
        enhanced: Enhanced image
        save_path: Optional save path
    
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Normalize
    if original.max() <= 1.0:
        original = (original * 255).astype(np.uint8)
    if enhanced.max() <= 1.0:
        enhanced = (enhanced * 255).astype(np.uint8)
    
    # Handle channel-first
    if original.ndim == 3 and original.shape[0] == 3:
        original = np.transpose(original, (1, 2, 0))
    if enhanced.ndim == 3 and enhanced.shape[0] == 3:
        enhanced = np.transpose(enhanced, (1, 2, 0))
    
    colors = ['red', 'green', 'blue']
    channel_names = ['Red', 'Green', 'Blue']
    
    for i, (color, name) in enumerate(zip(colors, channel_names)):
        # Original histogram
        axes[0, i].hist(original[:, :, i].ravel(), bins=256, color=color, alpha=0.7, range=(0, 256))
        axes[0, i].set_title(f'{name} Channel - Original', fontsize=10, fontweight='bold')
        axes[0, i].set_xlim([0, 255])
        
        # Enhanced histogram
        axes[1, i].hist(enhanced[:, :, i].ravel(), bins=256, color=color, alpha=0.7, range=(0, 256))
        axes[1, i].set_title(f'{name} Channel - Enhanced', fontsize=10, fontweight='bold')
        axes[1, i].set_xlim([0, 255])
    
    plt.suptitle('Color Channel Distribution Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig
