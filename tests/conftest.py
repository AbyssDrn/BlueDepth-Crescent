"""
Pytest configuration and shared fixtures
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil
from PIL import Image

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import UNetStandard, UNetLight, UNetAttention, UnderwaterClassifier
from training.dataset import UnderwaterDataset
from training.device_manager import DeviceManager


# ============================================================================
# SESSION FIXTURES (Run once per test session)
# ============================================================================

@pytest.fixture(scope="session")
def test_device():
    """Get test device (CPU for consistency)"""
    return torch.device('cpu')


@pytest.fixture(scope="session")
def temp_dir():
    """Create temporary directory for tests"""
    temp_path = Path(tempfile.mkdtemp(prefix='bluedepth_test_'))
    yield temp_path
    # Cleanup after all tests
    if temp_path.exists():
        shutil.rmtree(temp_path)


@pytest.fixture(scope="session")
def sample_image_path(temp_dir):
    """Create a sample image for testing"""
    img_dir = temp_dir / "images"
    img_dir.mkdir(exist_ok=True)
    
    # Create dummy image
    img = Image.new('RGB', (256, 256), color='blue')
    img_path = img_dir / "test_image.jpg"
    img.save(img_path)
    
    return img_path


# ============================================================================
# FUNCTION FIXTURES (Run for each test)
# ============================================================================

@pytest.fixture
def dummy_tensor():
    """Create dummy input tensor"""
    return torch.randn(1, 3, 256, 256)


@pytest.fixture
def dummy_batch():
    """Create dummy batch of tensors"""
    return torch.randn(4, 3, 256, 256)


@pytest.fixture
def dummy_image_array():
    """Create dummy numpy image array"""
    return np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)


@pytest.fixture
def unet_light_model(test_device):
    """Initialize UNet Light model"""
    model = UNetLight()
    model.to(test_device)
    model.eval()
    return model


@pytest.fixture
def unet_standard_model(test_device):
    """Initialize UNet Standard model"""
    model = UNetStandard()
    model.to(test_device)
    model.eval()
    return model


@pytest.fixture
def unet_attention_model(test_device):
    """Initialize UNet Attention model"""
    model = UNetAttention()
    model.to(test_device)
    model.eval()
    return model


@pytest.fixture
def classifier_model(test_device):
    """Initialize Classifier model"""
    model = UnderwaterClassifier(num_classes=10)
    model.to(test_device)
    model.eval()
    return model


@pytest.fixture
def mock_dataset(temp_dir):
    """Create mock dataset for testing"""
    hazy_dir = temp_dir / "hazy"
    clear_dir = temp_dir / "clear"
    hazy_dir.mkdir(exist_ok=True)
    clear_dir.mkdir(exist_ok=True)
    
    # Create 5 dummy image pairs
    for i in range(5):
        hazy_img = Image.new('RGB', (256, 256), color='gray')
        clear_img = Image.new('RGB', (256, 256), color='white')
        
        hazy_img.save(hazy_dir / f"img_{i}.jpg")
        clear_img.save(clear_dir / f"img_{i}.jpg")
    
    return hazy_dir, clear_dir


@pytest.fixture
def mock_checkpoint(temp_dir, unet_standard_model):
    """Create mock checkpoint file"""
    checkpoint_path = temp_dir / "mock_checkpoint.pth"
    
    checkpoint = {
        'model_state_dict': unet_standard_model.state_dict(),
        'epoch': 10,
        'train_loss': 0.5,
        'val_loss': 0.6,
        'psnr': 28.5,
        'ssim': 0.92
    }
    
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


# ============================================================================
# PARAMETRIZE HELPERS
# ============================================================================

def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests requiring GPU"
    )
    config.addinivalue_line(
        "markers", "integration: marks integration tests"
    )


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def assert_tensor_shape(tensor, expected_shape):
    """Assert tensor has expected shape"""
    assert tensor.shape == expected_shape, f"Expected shape {expected_shape}, got {tensor.shape}"


def assert_tensor_range(tensor, min_val=0.0, max_val=1.0):
    """Assert tensor values are in expected range"""
    assert tensor.min() >= min_val, f"Tensor min {tensor.min()} < {min_val}"
    assert tensor.max() <= max_val, f"Tensor max {tensor.max()} > {max_val}"


def assert_model_trainable(model):
    """Assert model has trainable parameters"""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable_params > 0, "Model has no trainable parameters"
