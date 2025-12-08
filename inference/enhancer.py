"""
Underwater Image Enhancement Inference Module
Supports multiple UNet architectures for maritime security
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
import torchvision.transforms as transforms
from typing import Union, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class ImageEnhancer:
    """
    Underwater Image Enhancer
    
    Removes turbidity, haziness, color distortions, and improves clarity
    for maritime security and reconnaissance applications.
    """
    
    def __init__(
        self, 
        model_path: str,
        model_type: str = 'unet_standard',
        device: str = 'cuda'
    ):
        """
        Initialize the image enhancer
        
        Args:
            model_path: Path to trained model checkpoint
            model_type: Model architecture ('unet_standard', 'unet_light', 'unet_attention')
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type
        
        # Load appropriate model
        self.model = self._load_model(model_path, model_type)
        self.model.to(self.device)
        self.model.eval()
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        self.denorm = transforms.ToPILImage()
        
        logger.info(f"ImageEnhancer initialized with {model_type} on {self.device}")
    
    def _load_model(self, model_path: str, model_type: str):
        """Load the specified model architecture"""
        from models import UNetStandard, UNetLight, UNetAttention
        
        # Select model architecture
        if model_type == 'unet_standard':
            model = UNetStandard(n_channels=3, n_classes=3)
        elif model_type == 'unet_light':
            model = UNetLight(n_channels=3, n_classes=3)
        elif model_type == 'unet_attention':
            model = UNetAttention(n_channels=3, n_classes=3)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        logger.info(f"Loaded {model_type} from {model_path}")
        return model
    
    @torch.no_grad()
    def enhance_image(
        self, 
        image: Union[str, Image.Image, np.ndarray],
        return_tensor: bool = False
    ) -> Union[Image.Image, torch.Tensor]:
        """
        Enhance a single underwater image
        
        Removes:
        - Turbidity and haziness
        - Bluish/greenish color wavelength dominance
        - Illumination issues
        - Blurriness
        
        Enhances:
        - Color correction and white balance
        - Clarity and sharpness
        - Border definition
        
        Args:
            image: Input image (path, PIL Image, or numpy array)
            return_tensor: If True, return torch.Tensor instead of PIL Image
            
        Returns:
            Enhanced image
        """
        # Load and convert image
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        original_size = image.size
        
        # Transform to tensor
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Inference
        output_tensor = self.model(input_tensor)
        
        if return_tensor:
            return output_tensor
        
        # Convert back to PIL Image
        output_image = self.denorm(output_tensor.squeeze(0).cpu())
        output_image = output_image.resize(original_size, Image.LANCZOS)
        
        return output_image
    
    def enhance_batch(
        self, 
        image_dir: str, 
        output_dir: str,
        save_comparisons: bool = False
    ) -> Dict[str, int]:
        """
        Enhance all images in a directory
        
        Args:
            image_dir: Directory containing input images
            output_dir: Directory to save enhanced images
            save_comparisons: If True, save side-by-side comparisons
            
        Returns:
            Dictionary with processing statistics
        """
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all images
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        images = []
        for ext in image_extensions:
            images.extend(list(image_dir.glob(ext)))
            images.extend(list(image_dir.glob(ext.upper())))
        
        processed = 0
        failed = 0
        
        for img_path in images:
            try:
                # Enhance image
                enhanced = self.enhance_image(str(img_path))
                
                # Save enhanced image
                output_path = output_dir / img_path.name
                enhanced.save(output_path, quality=95)
                
                # Optional: Save comparison
                if save_comparisons:
                    self._save_comparison(img_path, enhanced, output_dir)
                
                processed += 1
                logger.info(f"Enhanced: {img_path.name}")
                
            except Exception as e:
                failed += 1
                logger.error(f"Failed to enhance {img_path.name}: {e}")
        
        stats = {
            'total': len(images),
            'processed': processed,
            'failed': failed
        }
        
        logger.info(f"Batch processing complete: {stats}")
        return stats
    
    def _save_comparison(
        self, 
        original_path: Path, 
        enhanced: Image.Image, 
        output_dir: Path
    ):
        """Save side-by-side comparison of original and enhanced images"""
        comparison_dir = output_dir / 'comparisons'
        comparison_dir.mkdir(exist_ok=True)
        
        original = Image.open(original_path).convert('RGB')
        
        # Create side-by-side comparison
        width, height = original.size
        comparison = Image.new('RGB', (width * 2, height))
        comparison.paste(original, (0, 0))
        comparison.paste(enhanced, (width, 0))
        
        comparison_path = comparison_dir / f"comparison_{original_path.name}"
        comparison.save(comparison_path, quality=95)
    
    def get_model_info(self) -> Dict[str, any]:
        """Get information about the loaded model"""
        return {
            'model_type': self.model_type,
            'device': str(self.device),
            'parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_params': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }
