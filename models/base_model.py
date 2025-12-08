"""
Enhanced Base Model Class for BlueDepth-Crescent
Multi-GPU support with automatic optimization
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging
from pathlib import Path
import platform

logger = logging.getLogger(__name__)


class BaseModel(nn.Module, ABC):
    """
    Enhanced base class for all BlueDepth-Crescent models
    
    Features:
    - Multi-GPU support (Desktop, Laptop, Jetson)
    - Automatic device optimization
    - Checkpoint management
    - Layer freezing for transfer learning
    - Comprehensive model info
    """
    
    def __init__(self, name: str = "BaseModel"):
        super().__init__()
        self.name = name
        self.device = torch.device('cpu')
        self.training_history = []
        self.device_type = None  # Will be set by to_device()
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass - must be implemented by subclasses"""
        pass
    
    def to_device(self, device: torch.device):
        """
        Move model to specified device with automatic optimization
        
        Supports:
        - Desktop GPUs (RTX 3060, 3070, 4070, etc.)
        - Laptop GPUs (RTX 4050, 3050, etc.)
        - Edge Devices (Jetson Nano, Jetson Orin)
        - CPU fallback
        
        Args:
            device: Target device (cuda or cpu)
            
        Returns:
            Self for method chaining
        """
        self.device = device
        self.to(device)
        
        # Detect device type
        self.device_type = self._detect_device_type()
        
        # Apply device-specific optimizations
        if device.type == 'cuda':
            self._optimize_for_gpu()
            logger.info(f"{self.name} moved to {self.device_type} with optimizations")
        else:
            logger.info(f"{self.name} moved to CPU")
        
        return self
    
    def _detect_device_type(self) -> str:
        """Detect the type of device we're running on"""
        if self.device.type != 'cuda':
            return "CPU"
        
        try:
            gpu_name = torch.cuda.get_device_name(0).lower()
            platform_info = platform.platform().lower()
            
            # Jetson devices
            if 'jetson' in platform_info or 'tegra' in gpu_name:
                if 'nano' in gpu_name or 'tegra' in gpu_name:
                    return "JETSON_NANO"
                elif 'orin' in gpu_name:
                    return "JETSON_ORIN"
                elif 'xavier' in gpu_name:
                    return "JETSON_XAVIER"
                else:
                    return "JETSON"
            
            # Desktop vs Laptop GPUs
            if 'mobile' in gpu_name or 'laptop' in gpu_name or 'max-q' in gpu_name:
                return "LAPTOP_GPU"
            else:
                return "DESKTOP_GPU"
                
        except Exception as e:
            logger.warning(f"Could not detect device type: {e}")
            return "UNKNOWN_GPU"
    
    def _optimize_for_gpu(self):
        """Apply GPU-specific optimizations"""
        try:
            # Enable cudnn benchmarking for better performance
            torch.backends.cudnn.benchmark = True
            logger.info("cuDNN benchmark mode enabled")
            
            # Jetson-specific optimizations
            if self.device_type and self.device_type.startswith("JETSON"):
                logger.info("Applying Jetson-specific optimizations")
                
                # Enable TF32 for better performance on newer Jetson (Orin)
                cuda_capability = torch.cuda.get_device_capability(0)
                if cuda_capability[0] >= 8:
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                    logger.info("TF32 acceleration enabled for Jetson Orin")
                
                # Set memory management for shared memory systems
                if self.device_type == "JETSON_NANO":
                    logger.info("Conservative memory management for Jetson Nano")
            
            # Desktop/Laptop GPU optimizations
            else:
                # Enable TF32 on Ampere and newer (RTX 3000+, RTX 4000+)
                cuda_capability = torch.cuda.get_device_capability(0)
                if cuda_capability[0] >= 8:
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                    logger.info("TF32 acceleration enabled")
        
        except Exception as e:
            logger.warning(f"Could not apply all GPU optimizations: {e}")
    
    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def count_all_parameters(self) -> int:
        """Count all parameters (trainable and non-trainable)"""
        return sum(p.numel() for p in self.parameters())
    
    def get_model_size_mb(self) -> float:
        """Get model size in megabytes"""
        param_size = 0
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
        
        buffer_size = 0
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
        info = {
            'name': self.name,
            'trainable_params': self.count_parameters(),
            'total_params': self.count_all_parameters(),
            'size_mb': self.get_model_size_mb(),
            'device': str(self.device),
            'device_type': self.device_type,
            'is_training': self.training
        }
        
        # Add GPU-specific info
        if self.device.type == 'cuda':
            try:
                info['gpu_name'] = torch.cuda.get_device_name(0)
                info['cuda_capability'] = torch.cuda.get_device_capability(0)
            except:
                pass
        
        return info
    
    def print_model_info(self):
        """Print model information in formatted way"""
        info = self.get_model_info()
        print("\n" + "="*60)
        print(f"Model: {info['name']}")
        print("="*60)
        print(f"Trainable Parameters: {info['trainable_params']:,}")
        print(f"Total Parameters:     {info['total_params']:,}")
        print(f"Model Size:           {info['size_mb']:.2f} MB")
        print(f"Device:               {info['device']}")
        if info.get('device_type'):
            print(f"Device Type:          {info['device_type']}")
        if info.get('gpu_name'):
            print(f"GPU:                  {info['gpu_name']}")
        print(f"Training Mode:        {info['is_training']}")
        print("="*60 + "\n")
    
    def save_checkpoint(
        self, 
        path: str, 
        optimizer=None, 
        epoch: int = 0,
        loss: float = 0.0, 
        metrics: Optional[Dict] = None, 
        **kwargs
    ):
        """Save comprehensive model checkpoint"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'model_name': self.name,
            'model_info': self.get_model_info(),
            'epoch': epoch,
            'loss': loss,
            'metrics': metrics or {},
            **kwargs
        }
        
        if optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
        logger.info(f"Epoch: {epoch}, Loss: {loss:.6f}")
    
    def load_checkpoint(
        self, 
        path: str, 
        optimizer=None, 
        device: Optional[torch.device] = None
    ) -> Dict:
        """Load model checkpoint"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        if device is None:
            device = self.device
        
        checkpoint = torch.load(path, map_location=device)
        
        # Load model state
        self.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if provided
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        logger.info(f"Checkpoint loaded from {path}")
        if 'epoch' in checkpoint:
            logger.info(f"Restored epoch: {checkpoint['epoch']}")
        if 'loss' in checkpoint:
            logger.info(f"Checkpoint loss: {checkpoint['loss']:.6f}")
        
        return checkpoint
    
    def freeze_layers(self, layer_names: Optional[list] = None):
        """Freeze specific layers or all layers"""
        if layer_names is None:
            # Freeze all layers
            for param in self.parameters():
                param.requires_grad = False
            logger.info(f"All layers frozen in {self.name}")
        else:
            # Freeze specific layers
            for name, param in self.named_parameters():
                if any(layer_name in name for layer_name in layer_names):
                    param.requires_grad = False
                    logger.info(f"Frozen layer: {name}")
    
    def unfreeze_layers(self, layer_names: Optional[list] = None):
        """Unfreeze specific layers or all layers"""
        if layer_names is None:
            # Unfreeze all layers
            for param in self.parameters():
                param.requires_grad = True
            logger.info(f"All layers unfrozen in {self.name}")
        else:
            # Unfreeze specific layers
            for name, param in self.named_parameters():
                if any(layer_name in name for layer_name in layer_names):
                    param.requires_grad = True
                    logger.info(f"Unfrozen layer: {name}")
    
    def get_layer_names(self) -> list:
        """Get list of all layer names"""
        return [name for name, _ in self.named_parameters()]
    
    def initialize_weights(self):
        """Initialize weights using He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        logger.info(f"Weights initialized for {self.name}")
    
    def set_train_mode(self):
        """Set model to training mode"""
        self.train()
        logger.info(f"{self.name} set to training mode")
    
    def set_eval_mode(self):
        """Set model to evaluation mode"""
        self.eval()
        logger.info(f"{self.name} set to evaluation mode")
    
    def get_trainable_layers(self) -> list:
        """Get list of trainable layers"""
        return [name for name, param in self.named_parameters() if param.requires_grad]
    
    def get_frozen_layers(self) -> list:
        """Get list of frozen layers"""
        return [name for name, param in self.named_parameters() if not param.requires_grad]
    
    def optimize_for_inference(self):
        """Optimize model for inference (no gradients, eval mode)"""
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
        logger.info(f"{self.name} optimized for inference")
