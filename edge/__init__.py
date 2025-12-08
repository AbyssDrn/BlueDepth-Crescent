"""
BlueDepth-Crescent Edge Deployment Module
Optimized deployment for Jetson Nano/Xavier/Orin and embedded systems
"""

__version__ = "1.0.0"
__author__ = "BlueDepth Team"

# Export main classes
from .jetson_inference import JetsonInference
from .export_onnx import export_to_onnx
from .convert_trt import convert_to_tensorrt
from .optimize_model import prune_model, quantize_model

__all__ = [
    'JetsonInference',
    'export_to_onnx',
    'convert_to_tensorrt',
    'prune_model',
    'quantize_model'
]

# Platform detection
import platform
import sys

def get_platform_info():
    """Get current platform information"""
    return {
        'system': platform.system(),
        'machine': platform.machine(),
        'python_version': sys.version,
        'is_jetson': platform.machine() in ['aarch64', 'arm64']
    }

PLATFORM_INFO = get_platform_info()

# Check if running on Jetson
IS_JETSON = PLATFORM_INFO['is_jetson']

if IS_JETSON:
    print(" Running on Jetson platform")
else:
    print(" Not running on Jetson - some features may not be available")
