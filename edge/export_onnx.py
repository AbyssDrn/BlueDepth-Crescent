#!/usr/bin/env python3
"""
Export PyTorch Models to ONNX Format
Supports all BlueDepth-Crescent model architectures
"""

import sys
import torch
import logging
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import UNetLight, UNetStandard, UNetAttention, UnderwaterClassifier
from utils.logger import setup_logger

logger = setup_logger(__name__)


def export_to_onnx(
    model_path: str,
    output_path: str,
    model_type: str = 'unet_standard',
    img_size: int = 256,
    opset_version: int = 13,
    dynamic_axes: bool = True
):
    """
    Export PyTorch model to ONNX format
    
    Args:
        model_path: Path to PyTorch checkpoint (.pth)
        output_path: Path to save ONNX model (.onnx)
        model_type: Type of model ('unet_light', 'unet_standard', 'unet_attention', 'classifier')
        img_size: Input image size
        opset_version: ONNX opset version
        dynamic_axes: Enable dynamic batch/spatial dimensions
    """
    
    logger.info(f"Exporting {model_type} to ONNX...")
    logger.info(f"Source: {model_path}")
    logger.info(f"Output: {output_path}")
    
    # Select model architecture
    if model_type == 'unet_light':
        model = UNetLight()
    elif model_type == 'unet_standard':
        model = UNetStandard()
    elif model_type == 'unet_attention':
        model = UNetAttention()
    elif model_type == 'classifier':
        model = UnderwaterClassifier(num_classes=10)
        img_size = 224  # Classifier uses 224x224
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load checkpoint
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    model.to(device)
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, img_size, img_size).to(device)
    
    # Test forward pass
    with torch.no_grad():
        test_output = model(dummy_input)
    logger.info(f"Test forward pass successful: {test_output.shape}")
    
    # Prepare dynamic axes
    if dynamic_axes:
        dynamic_config = {
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size', 2: 'height', 3: 'width'}
        }
    else:
        dynamic_config = None
    
    # Export to ONNX
    logger.info("Converting to ONNX...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_config,
        verbose=False
    )
    
    logger.info(f" Model exported to ONNX: {output_path}")
    
    # Verify ONNX model
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        logger.info(" ONNX model verified successfully")
        
        # Print model info
        logger.info(f"ONNX opset version: {onnx_model.opset_import[0].version}")
        logger.info(f"Input: {onnx_model.graph.input[0].name}")
        logger.info(f"Output: {onnx_model.graph.output[0].name}")
        
    except ImportError:
        logger.warning(" ONNX not installed - skipping verification")
    except Exception as e:
        logger.error(f" ONNX verification failed: {e}")
        raise
    
    # Get file size
    file_size = Path(output_path).stat().st_size / (1024 * 1024)  # MB
    logger.info(f"ONNX model size: {file_size:.2f} MB")
    
    return output_path


def export_all_models(checkpoint_dir: str = "checkpoints", output_dir: str = "edge/models"):
    """Export all trained models to ONNX"""
    
    checkpoint_dir = Path(checkpoint_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    models_config = [
        {
            'name': 'UNet-Light',
            'type': 'unet_light',
            'checkpoint': checkpoint_dir / 'unet_light_best.pth',
            'output': output_dir / 'unet_light.onnx'
        },
        {
            'name': 'UNet-Standard',
            'type': 'unet_standard',
            'checkpoint': checkpoint_dir / 'unet_standard_best.pth',
            'output': output_dir / 'unet_standard.onnx'
        },
        {
            'name': 'UNet-Attention',
            'type': 'unet_attention',
            'checkpoint': checkpoint_dir / 'unet_attention_best.pth',
            'output': output_dir / 'unet_attention.onnx'
        },
        {
            'name': 'Classifier',
            'type': 'classifier',
            'checkpoint': checkpoint_dir / 'classifier_best.pth',
            'output': output_dir / 'classifier.onnx'
        }
    ]
    
    for config in models_config:
        if config['checkpoint'].exists():
            logger.info(f"\n{'='*60}")
            logger.info(f"Exporting {config['name']}")
            logger.info(f"{'='*60}")
            
            try:
                export_to_onnx(
                    model_path=str(config['checkpoint']),
                    output_path=str(config['output']),
                    model_type=config['type']
                )
            except Exception as e:
                logger.error(f"Failed to export {config['name']}: {e}")
        else:
            logger.warning(f"Checkpoint not found: {config['checkpoint']}")


def main():
    """Main export function"""
    parser = argparse.ArgumentParser(description="Export PyTorch models to ONNX")
    parser.add_argument('--model-path', type=str, help="Path to PyTorch checkpoint")
    parser.add_argument('--output-path', type=str, help="Path to save ONNX model")
    parser.add_argument('--model-type', type=str, default='unet_standard',
                       choices=['unet_light', 'unet_standard', 'unet_attention', 'classifier'],
                       help="Model architecture type")
    parser.add_argument('--img-size', type=int, default=256, help="Input image size")
    parser.add_argument('--opset-version', type=int, default=13, help="ONNX opset version")
    parser.add_argument('--export-all', action='store_true', help="Export all models")
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help="Directory containing checkpoints")
    parser.add_argument('--output-dir', type=str, default='edge/models',
                       help="Directory to save ONNX models")
    
    args = parser.parse_args()
    
    if args.export_all:
        export_all_models(args.checkpoint_dir, args.output_dir)
    else:
        if not args.model_path or not args.output_path:
            parser.error("--model-path and --output-path required when not using --export-all")
        
        export_to_onnx(
            model_path=args.model_path,
            output_path=args.output_path,
            model_type=args.model_type,
            img_size=args.img_size,
            opset_version=args.opset_version
        )


if __name__ == "__main__":
    main()
