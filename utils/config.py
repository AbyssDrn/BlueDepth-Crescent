"""
Configuration Management for BlueDepth-Crescent
Centralized paths and settings aligned with organized data structure
"""

import yaml
import os
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field

# Project Root
PROJECT_ROOT = Path(__file__).parent.parent

# Data Paths (aligned with organize_data.py structure)
DATA_ROOT = PROJECT_ROOT / "data"

@dataclass
class DataPaths:
    """Organized data folder paths"""
    # Training data
    train_root: Path = DATA_ROOT / "train"
    train_hazy: Path = DATA_ROOT / "train" / "hazy"
    train_clear: Path = DATA_ROOT / "train" / "clear"
    
    # Testing data
    test_root: Path = DATA_ROOT / "test"
    test_hazy: Path = DATA_ROOT / "test" / "hazy"
    test_clear: Path = DATA_ROOT / "test" / "clear"
    
    # Backup and processing
    raw: Path = DATA_ROOT / "raw"
    enhanced: Path = DATA_ROOT / "enhanced"
    processed: Path = DATA_ROOT / "processed"
    
    # Video data
    videos: Path = DATA_ROOT / "videos"
    frames: Path = DATA_ROOT / "frames"

@dataclass
class ModelPaths:
    """Model-related paths"""
    checkpoints: Path = PROJECT_ROOT / "checkpoints"
    logs: Path = PROJECT_ROOT / "logs"
    tensorboard: Path = PROJECT_ROOT / "runs"
    
    def get_checkpoint(self, model_type: str, name: str = "best") -> Path:
        """Get checkpoint path for specific model"""
        return self.checkpoints / f"{model_type}_{name}.pth"

@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    # Model settings
    model_type: str = "unet_attention"  # 'unet_standard', 'unet_light', 'unet_attention'
    img_size: int = 256
    
    # Training parameters
    batch_size: int = 4
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    
    # Loss weights
    l1_weight: float = 1.0
    perceptual_weight: float = 0.1
    ssim_weight: float = 0.5
    
    # Optimization
    use_mixed_precision: bool = True
    gradient_clip: float = 1.0
    
    # Data augmentation
    augment: bool = True
    
    # Device management
    device: str = "cuda"  # 'cuda' or 'cpu'
    num_workers: int = 4

@dataclass
class InferenceConfig:
    """Inference settings"""
    device: str = "cuda"
    batch_size: int = 1
    save_comparisons: bool = True
    output_quality: int = 95  # JPEG quality

@dataclass
class VideoConfig:
    """Video processing settings"""
    fps: int = 5  # Frames per second to extract
    output_fps: int = 30  # Output video FPS
    codec: str = 'mp4v'  # Video codec
    frame_format: str = 'jpg'  # Frame image format

class Config:
    """
    Centralized configuration management
    Manages all paths, hyperparameters, and settings
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration
        
        Args:
            config_path: Optional YAML config file path
        """
        self.data = DataPaths()
        self.model = ModelPaths()
        self.training = TrainingConfig()
        self.inference = InferenceConfig()
        self.video = VideoConfig()
        
        # Load from YAML if provided
        if config_path:
            self.load_yaml(config_path)
        
        # Create directories
        self._create_directories()
    
    def load_yaml(self, config_path: str) -> None:
        """Load configuration from YAML file"""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update configurations
        if 'training' in config:
            for key, value in config['training'].items():
                if hasattr(self.training, key):
                    setattr(self.training, key, value)
        
        if 'inference' in config:
            for key, value in config['inference'].items():
                if hasattr(self.inference, key):
                    setattr(self.inference, key, value)
    
    def save_yaml(self, path: str) -> None:
        """Save current configuration to YAML"""
        config = {
            'training': self.training.__dict__,
            'inference': self.inference.__dict__,
            'video': self.video.__dict__
        }
        
        with open(path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    def _create_directories(self) -> None:
        """Create necessary directories"""
        dirs_to_create = [
            self.data.train_hazy,
            self.data.train_clear,
            self.data.test_hazy,
            self.data.test_clear,
            self.data.enhanced,
            self.data.processed,
            self.data.frames,
            self.data.videos,
            self.data.raw,
            self.model.checkpoints,
            self.model.logs,
            self.model.tensorboard
        ]
        
        for dir_path in dirs_to_create:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        Example: config.get('training.batch_size')
        """
        keys = key.split('.')
        obj = self
        
        for k in keys:
            if hasattr(obj, k):
                obj = getattr(obj, k)
            else:
                return default
        
        return obj
    
    def __repr__(self) -> str:
        return f"Config(model={self.training.model_type}, batch_size={self.training.batch_size})"


# Global configuration instance
PROJECT_CONFIG = Config()

# Environment variables override
if os.getenv('BLUEDEPTH_DATA_ROOT'):
    DATA_ROOT = Path(os.getenv('BLUEDEPTH_DATA_ROOT'))

if os.getenv('BLUEDEPTH_DEVICE'):
    PROJECT_CONFIG.training.device = os.getenv('BLUEDEPTH_DEVICE')
