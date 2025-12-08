"""
BlueDepth-Crescent Inference Module
Underwater image enhancement and object classification for maritime security
"""

from .enhancer import ImageEnhancer
from .classifier import ObjectClassifier, EnhancedClassifier
from .batch_processor import BatchProcessor, VideoFrameProcessor
from .video_processor import VideoProcessor

__all__ = [
    'ImageEnhancer',
    'ObjectClassifier',
    'EnhancedClassifier',
    'BatchProcessor',
    'VideoFrameProcessor',
    'VideoProcessor'
]
