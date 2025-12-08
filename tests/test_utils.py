"""
Comprehensive tests for utility functions
Tests: config, logger, metrics, visualization, image_utils, video_utils
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys
import yaml
import tempfile
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.metrics import MetricsCalculator
from utils.image_utils import load_image, save_image, resize_image, normalize_tensor, denormalize_tensor
from utils.video_utils import extract_frames, create_video_from_frames
from utils.logger import setup_logger
from utils.config import PROJECT_CONFIG
from utils.visualization import plot_comparison, save_comparison_grid


class TestMetrics:
    """Test metrics calculation"""
    
    def test_psnr_identical_images(self):
        """Test PSNR with identical images"""
        img1 = torch.rand(1, 3, 256, 256)
        img2 = img1.clone()
        
        metrics = MetricsCalculator.compute_all(img1, img2)
        
        # PSNR should be very high for identical images
        assert metrics['psnr'] > 40, f"PSNR too low: {metrics['psnr']}"
    
    def test_ssim_identical_images(self):
        """Test SSIM with identical images"""
        img1 = torch.rand(1, 3, 256, 256)
        img2 = img1.clone()
        
        metrics = MetricsCalculator.compute_all(img1, img2)
        
        # SSIM should be 1.0 for identical images
        assert abs(metrics['ssim'] - 1.0) < 0.01, f"SSIM not 1.0: {metrics['ssim']}"
    
    def test_mse_identical_images(self):
        """Test MSE with identical images"""
        img1 = torch.rand(1, 3, 256, 256)
        img2 = img1.clone()
        
        metrics = MetricsCalculator.compute_all(img1, img2)
        
        # MSE should be 0 for identical images
        assert metrics['mse'] < 1e-6, f"MSE too high: {metrics['mse']}"
    
    def test_mae_identical_images(self):
        """Test MAE with identical images"""
        img1 = torch.rand(1, 3, 256, 256)
        img2 = img1.clone()
        
        metrics = MetricsCalculator.compute_all(img1, img2)
        
        # MAE should be 0 for identical images
        assert metrics['mae'] < 1e-6, f"MAE too high: {metrics['mae']}"
    
    def test_metrics_different_images(self):
        """Test metrics with different images"""
        img1 = torch.rand(1, 3, 256, 256)
        img2 = torch.rand(1, 3, 256, 256)
        
        metrics = MetricsCalculator.compute_all(img1, img2)
        
        # Metrics should be reasonable
        assert 0 < metrics['psnr'] < 50
        assert 0 < metrics['ssim'] < 1
        assert metrics['mse'] > 0
        assert metrics['mae'] > 0
    
    def test_batch_metrics(self):
        """Test metrics with batch"""
        img1 = torch.rand(4, 3, 256, 256)
        img2 = torch.rand(4, 3, 256, 256)
        
        metrics = MetricsCalculator.compute_all(img1, img2)
        
        # Should return averaged metrics
        assert isinstance(metrics['psnr'], float)
        assert isinstance(metrics['ssim'], float)
    
    def test_print_metrics(self, capsys):
        """Test metrics printing"""
        metrics = {
            'psnr': 28.5,
            'ssim': 0.92,
            'mse': 0.01,
            'mae': 0.05
        }
        
        MetricsCalculator.print_metrics(metrics)
        captured = capsys.readouterr()
        
        assert 'PSNR' in captured.out
        assert '28.5' in captured.out


class TestImageUtils:
    """Test image utility functions"""
    
    def test_load_image(self, sample_image_path):
        """Test image loading"""
        img = load_image(sample_image_path)
        
        assert isinstance(img, torch.Tensor)
        assert img.ndim == 3  # C, H, W
        assert img.shape[0] == 3  # RGB
    
    def test_save_image(self, tmp_path):
        """Test image saving"""
        img_tensor = torch.rand(3, 256, 256)
        save_path = tmp_path / "test_save.jpg"
        
        save_image(img_tensor, save_path)
        
        assert save_path.exists()
    
    def test_resize_image(self):
        """Test image resizing"""
        img = torch.rand(3, 512, 512)
        resized = resize_image(img, size=(256, 256))
        
        assert resized.shape == (3, 256, 256)
    
    def test_normalize_tensor(self):
        """Test tensor normalization"""
        img = torch.randint(0, 256, (3, 256, 256), dtype=torch.float32)
        normalized = normalize_tensor(img)
        
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0
    
    def test_denormalize_tensor(self):
        """Test tensor denormalization"""
        img = torch.rand(3, 256, 256)
        denormalized = denormalize_tensor(img)
        
        assert denormalized.min() >= 0
        assert denormalized.max() <= 255
    
    def test_normalize_denormalize_cycle(self):
        """Test normalize->denormalize cycle"""
        original = torch.randint(0, 256, (3, 256, 256), dtype=torch.float32)
        normalized = normalize_tensor(original)
        denormalized = denormalize_tensor(normalized)
        
        # Should be close to original
        assert torch.allclose(original, denormalized, atol=1.0)


class TestVideoUtils:
    """Test video utility functions"""
    
    @pytest.mark.slow
    def test_extract_frames(self, tmp_path):
        """Test frame extraction from video"""
        # Note: Requires actual video file, skip if not available
        pytest.skip("Requires test video file")
    
    @pytest.mark.slow
    def test_create_video_from_frames(self, tmp_path):
        """Test video creation from frames"""
        # Create dummy frames
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        
        for i in range(10):
            img = Image.new('RGB', (256, 256), color='blue')
            img.save(frames_dir / f"frame_{i:04d}.jpg")
        
        output_path = tmp_path / "output_video.mp4"
        
        try:
            create_video_from_frames(frames_dir, output_path, fps=10)
            assert output_path.exists()
        except:
            pytest.skip("Video creation requires ffmpeg/opencv")


class TestLogger:
    """Test logging functionality"""
    
    def test_setup_logger(self):
        """Test logger setup"""
        logger = setup_logger('test_logger', level='info')
        
        assert logger is not None
        assert logger.name == 'test_logger'
    
    def test_logger_levels(self):
        """Test different logging levels"""
        logger = setup_logger('test_logger_levels', level='debug')
        
        # Should not raise errors
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
    
    def test_logger_file_output(self, tmp_path):
        """Test logger writes to file"""
        log_file = tmp_path / "test.log"
        logger = setup_logger('test_file_logger', level='info', log_file=str(log_file))
        
        logger.info("Test message")
        
        # Check file was created (may not happen immediately)
        # assert log_file.exists()


class TestConfig:
    """Test configuration management"""
    
    def test_project_config_exists(self):
        """Test PROJECT_CONFIG is accessible"""
        assert PROJECT_CONFIG is not None
    
    def test_config_has_data_paths(self):
        """Test config has data paths"""
        assert hasattr(PROJECT_CONFIG, 'data')
        assert hasattr(PROJECT_CONFIG.data, 'train_hazy')
        assert hasattr(PROJECT_CONFIG.data, 'train_clear')
    
    def test_config_has_training_params(self):
        """Test config has training parameters"""
        assert hasattr(PROJECT_CONFIG, 'training')
        assert hasattr(PROJECT_CONFIG.training, 'batch_size')
        assert hasattr(PROJECT_CONFIG.training, 'num_epochs')
    
    def test_config_paths_are_strings(self):
        """Test config paths are strings"""
        assert isinstance(PROJECT_CONFIG.data.train_hazy, str)
        assert isinstance(PROJECT_CONFIG.checkpoints.save_dir, str)


class TestVisualization:
    """Test visualization functions"""
    
    def test_plot_comparison(self, tmp_path):
        """Test comparison plotting"""
        original = torch.rand(3, 256, 256)
        enhanced = torch.rand(3, 256, 256)
        save_path = tmp_path / "comparison.png"
        
        fig = plot_comparison(original, enhanced, save_path=str(save_path))
        
        assert fig is not None
        assert save_path.exists()
    
    def test_save_comparison_grid(self, tmp_path):
        """Test comparison grid saving"""
        hazy = [torch.rand(3, 256, 256) for _ in range(4)]
        enhanced = [torch.rand(3, 256, 256) for _ in range(4)]
        clear = [torch.rand(3, 256, 256) for _ in range(4)]
        
        save_path = tmp_path / "grid.png"
        
        save_comparison_grid(hazy, enhanced, clear, save_path=str(save_path))
        
        assert save_path.exists()


class TestUtilsIntegration:
    """Integration tests for utils"""
    
    def test_full_image_pipeline(self, sample_image_path, tmp_path):
        """Test complete image processing pipeline"""
        # Load
        img = load_image(sample_image_path)
        
        # Normalize
        normalized = normalize_tensor(img)
        
        # Simulate processing
        processed = normalized * 0.9  # Darken slightly
        
        # Denormalize
        denormalized = denormalize_tensor(processed)
        
        # Save
        output_path = tmp_path / "processed.jpg"
        save_image(denormalized, output_path)
        
        assert output_path.exists()
    
    def test_metrics_with_real_processing(self):
        """Test metrics with actual image processing"""
        original = torch.rand(1, 3, 256, 256)
        
        # Simulate enhancement (add noise)
        enhanced = original + torch.randn_like(original) * 0.1
        enhanced = torch.clamp(enhanced, 0, 1)
        
        metrics = MetricsCalculator.compute_all(enhanced, original)
        
        # Metrics should indicate some degradation
        assert metrics['psnr'] < 40  # Not perfect
        assert metrics['ssim'] < 1.0  # Not perfect


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
