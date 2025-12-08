"""
BlueDepth-Crescent Test Suite
Comprehensive tests for all modules
"""

__version__ = "1.0.0"
__author__ = "BlueDepth Team"

# Test configuration
TEST_CONFIG = {
    'batch_size': 2,
    'img_size': 256,
    'num_workers': 0,  # 0 for testing
    'device': 'cpu',  # Use CPU for tests
    'tolerance': 1e-5
}

# Test data paths
TEST_DATA = {
    'temp_dir': 'tests/temp',
    'fixtures': 'tests/fixtures',
    'sample_images': 'tests/fixtures/images'
}

# Import test utilities
from .conftest import *

__all__ = [
    'TEST_CONFIG',
    'TEST_DATA'
]
