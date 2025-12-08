"""
Utility Functions for BlueDepth-Crescent
Maritime Security and Reconnaissance System
"""

from .config import Config, PROJECT_CONFIG
from .logger import setup_logger, get_logger
from .image_utils import (
    load_image, 
    save_image, 
    normalize, 
    denormalize,
    resize_image,
    create_comparison,
    calculate_image_stats
)
from .video_utils import (
    get_video_info,
    extract_frame,
    create_video_from_frames,
    VideoWriter
)
from .metrics import (
    calculate_psnr,
    calculate_ssim,
    calculate_mae,
    calculate_mse,
    MetricsCalculator,
    evaluate_enhancement
)
from .visualization import (
    plot_comparison,
    plot_training_curves,
    plot_metrics,
    save_comparison_grid,
    plot_loss_curves
)

__all__ = [
    # Config
    'Config',
    'PROJECT_CONFIG',
    
    # Logging
    'setup_logger',
    'get_logger',
    
    # Image utilities
    'load_image',
    'save_image',
    'normalize',
    'denormalize',
    'resize_image',
    'create_comparison',
    'calculate_image_stats',
    
    # Video utilities
    'get_video_info',
    'extract_frame',
    'create_video_from_frames',
    'VideoWriter',
    
    # Metrics
    'calculate_psnr',
    'calculate_ssim',
    'calculate_mae',
    'calculate_mse',
    'MetricsCalculator',
    'evaluate_enhancement',
    
    # Visualization
    'plot_comparison',
    'plot_training_curves',
    'plot_metrics',
    'save_comparison_grid',
    'plot_loss_curves'
]

__version__ = '1.0.0'
