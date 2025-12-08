"""BlueDepth-Crescent Models Package
Maritime Security and Reconnaissance System
"""

from .base_model import BaseModel
from .unet_standard import UNetStandard, DoubleConv, Down, Up
from .unet_light import UNetLight, DepthwiseSeparableConv
from .unet_attention import UNetAttention, AttentionBlock
from .classifier import (
    UnderwaterClassifier,
    OBJECT_CATEGORIES,
    THREAT_LEVELS,
    THREAT_COLORS,
    get_threat_info,
    get_threat_color
)

__version__ = '1.0.0'
__author__ = 'BlueDepth-Crescent Team'

__all__ = [
    # Base
    'BaseModel',
    
    # U-Net Models
    'UNetStandard',
    'UNetLight',
    'UNetAttention',
    
    # U-Net Components
    'DoubleConv',
    'Down',
    'Up',
    'DepthwiseSeparableConv',
    'AttentionBlock',
    
    # Classifier
    'UnderwaterClassifier',
    
    # Classifier Data
    'OBJECT_CATEGORIES',
    'THREAT_LEVELS',
    'THREAT_COLORS',
    
    # Classifier Functions
    'get_threat_info',
    'get_threat_color'
]

# Model registry for easy instantiation
MODEL_REGISTRY = {
    'unet_standard': UNetStandard,
    'unet_light': UNetLight,
    'unet_attention': UNetAttention,
    'classifier': UnderwaterClassifier
}

def get_model(model_name: str, **kwargs):
    """Factory function to get model by name
    
    Args:
        model_name: Name of model ('unet_standard', 'unet_light', 'unet_attention', 'classifier')
        **kwargs: Model-specific arguments
    
    Returns:
        Model instance
    
    Example:
        >>> model = get_model('unet_standard', n_channels=3, n_classes=3)
        >>> classifier = get_model('classifier', num_classes=15)
    """
    if model_name not in MODEL_REGISTRY:
        available = ', '.join(MODEL_REGISTRY.keys())
        raise ValueError(f"Model '{model_name}' not found. Available: {available}")
    
    return MODEL_REGISTRY[model_name](**kwargs)

def list_models():
    """List all available models"""
    return list(MODEL_REGISTRY.keys())

def get_model_info(model_name: str) -> dict:
    """Get information about a model
    
    Args:
        model_name: Name of model
    
    Returns:
        Dictionary with model information
    """
    model_info = {
        'unet_standard': {
            'name': 'UNet Standard',
            'description': 'Balanced underwater enhancement model',
            'parameters': '8.2M',
            'speed_rtx4050': '60-80 FPS',
            'use_case': 'General maritime operations'
        },
        'unet_light': {
            'name': 'UNet Light',
            'description': 'Lightweight model for edge deployment',
            'parameters': '2.1M',
            'speed_rtx4050': '120-150 FPS',
            'speed_jetson': '25-30 FPS',
            'use_case': 'Real-time ROV/AUV operations'
        },
        'unet_attention': {
            'name': 'UNet Attention',
            'description': 'Maximum quality enhancement',
            'parameters': '12.5M',
            'speed_rtx4050': '30-40 FPS',
            'use_case': 'Critical security analysis'
        },
        'classifier': {
            'name': 'Maritime Threat Classifier',
            'description': 'Object detection with threat assessment',
            'parameters': '5.8M',
            'speed_rtx4050': '320 FPS (batch=1)',
            'categories': 15,
            'threat_levels': 5,
            'use_case': 'Object identification and threat assessment'
        }
    }
    
    return model_info.get(model_name, {'error': 'Model not found'})