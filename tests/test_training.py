"""
Tests for training components
Tests: dataset, losses, device_manager
"""

import pytest
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.dataset import UnderwaterDataset
from training.losses import CombinedLoss, SSIMLoss
from training.device_manager import DeviceManager


class TestUnderwaterDataset:
    """Test dataset loading"""
    
    def test_dataset_initialization(self, mock_dataset):
        """Test dataset can be initialized"""
        hazy_dir, clear_dir = mock_dataset
        
        dataset = UnderwaterDataset(
            hazy_dir=str(hazy_dir),
            clear_dir=str(clear_dir),
            img_size=256
        )
        
        assert len(dataset) == 5
    
    def test_dataset_getitem(self, mock_dataset):
        """Test getting item from dataset"""
        hazy_dir, clear_dir = mock_dataset
        
        dataset = UnderwaterDataset(
            hazy_dir=str(hazy_dir),
            clear_dir=str(clear_dir),
            img_size=256
        )
        
        hazy, clear = dataset[0]
        
        assert hazy.shape == (3, 256, 256)
        assert clear.shape == (3, 256, 256)
    
    def test_dataset_augmentation(self, mock_dataset):
        """Test dataset with augmentation"""
        hazy_dir, clear_dir = mock_dataset
        
        dataset = UnderwaterDataset(
            hazy_dir=str(hazy_dir),
            clear_dir=str(clear_dir),
            img_size=256,
            augment=True
        )
        
        hazy1, _ = dataset[0]
        hazy2, _ = dataset[0]
        
        # Augmented images should be different
        assert not torch.equal(hazy1, hazy2)


class TestLosses:
    """Test loss functions"""
    
    def test_ssim_loss_identical(self):
        """Test SSIM loss with identical images"""
        loss_fn = SSIMLoss()
        
        img1 = torch.rand(1, 3, 256, 256)
        img2 = img1.clone()
        
        loss = loss_fn(img1, img2)
        
        # Loss should be near 0 for identical images
        assert loss < 0.01
    
    def test_ssim_loss_different(self):
        """Test SSIM loss with different images"""
        loss_fn = SSIMLoss()
        
        img1 = torch.rand(1, 3, 256, 256)
        img2 = torch.rand(1, 3, 256, 256)
        
        loss = loss_fn(img1, img2)
        
        # Loss should be > 0
        assert loss > 0
    
    def test_combined_loss(self):
        """Test combined loss function"""
        loss_fn = CombinedLoss(l1_weight=1.0, ssim_weight=0.5)
        
        img1 = torch.rand(1, 3, 256, 256)
        img2 = torch.rand(1, 3, 256, 256)
        
        loss = loss_fn(img1, img2)
        
        assert loss > 0
        assert not torch.isnan(loss)


class TestDeviceManager:
    """Test device management"""
    
    def test_device_manager_initialization(self):
        """Test device manager can be initialized"""
        manager = DeviceManager()
        
        assert manager.device is not None
    
    def test_device_is_valid(self):
        """Test device is valid"""
        manager = DeviceManager()
        
        assert manager.device.type in ['cpu', 'cuda', 'mps']
    
    def test_to_device(self):
        """Test moving tensors to device"""
        manager = DeviceManager()
        
        tensor = torch.rand(10, 10)
        device_tensor = manager.to_device(tensor)
        
        assert device_tensor.device == manager.device


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
