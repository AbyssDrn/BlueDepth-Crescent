"""Load best trained models"""

import torch
from pathlib import Path
from typing import Dict
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models import UNetLight, UNetStandard, UNetAttention


def get_best_models() -> Dict[str, dict]:
    """
    Scan checkpoints directory and return best models
    
    Returns:
        dict: {model_name: model_info}
    """
    
    checkpoint_dir = Path("checkpoints")
    
    if not checkpoint_dir.exists():
        print(f"Warning: Checkpoint directory not found: {checkpoint_dir}")
        return {}
    
    models_info = {}
    
    # Model configurations
    model_configs = {
        'unet_light': {'architecture': 'UNet-Light', 'params': '350K'},
        'unet_standard': {'architecture': 'UNet-Standard', 'params': '7.8M'},
        'unet_attention': {'architecture': 'UNet-Attention', 'params': '8.2M'}
    }
    
    # Find best checkpoints
    for model_name, config in model_configs.items():
        best_checkpoint = checkpoint_dir / f"{model_name}_best.pth"
        
        if best_checkpoint.exists():
            try:
                # Load checkpoint metadata
                checkpoint = torch.load(best_checkpoint, map_location='cpu')
                
                models_info[model_name] = {
                    'path': str(best_checkpoint),
                    'architecture': config['architecture'],
                    'parameters': config['params'],
                    'best_psnr': checkpoint.get('best_psnr', 0),
                    'epochs': checkpoint.get('epoch', 'Unknown'),
                    'training_history': checkpoint.get('training_history', {})
                }
                
            except Exception as e:
                print(f"Error loading {model_name}: {e}")
    
    return models_info
