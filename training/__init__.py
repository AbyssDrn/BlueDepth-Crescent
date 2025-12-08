"""
BlueDepth-Crescent Training Module
Underwater Image Enhancement Training Pipeline
Maritime Security and Reconnaissance System
"""

from .train_unet import UNetTrainer
from .dataset import UnderwaterDataset, VideoFrameDataset, ClassificationDataset
from .losses import (
    CombinedLoss, 
    SSIMLoss, 
    PerceptualLoss,
    UnderwaterColorLoss,
    EdgePreservationLoss,
    ClassificationLoss
)
from .device_manager import DeviceManager

__all__ = [
    'UNetTrainer',
    'UnderwaterDataset',
    'VideoFrameDataset',
    'ClassificationDataset',
    'CombinedLoss',
    'SSIMLoss',
    'PerceptualLoss',
    'UnderwaterColorLoss',
    'EdgePreservationLoss',
    'ClassificationLoss',
    'DeviceManager'
]
