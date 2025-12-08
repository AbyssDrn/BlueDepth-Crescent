#!/usr/bin/env python3
"""
Model Optimization for Edge Deployment
Includes pruning, quantization, and compression techniques
"""

import sys
import torch
import torch.nn as nn
import logging
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import UNetLight, UNetStandard, UNetAttention
from utils.logger import setup_logger

logger = setup_logger(__name__)


def prune_model(model: nn.Module, amount: float = 0.3) -> nn.Module:
    """
    Prune model weights using L1 unstructured pruning
    
    Args:
        model: PyTorch model
        amount: Fraction of weights to prune (0.0-1.0)
    
    Returns:
        Pruned model
    """
    import torch.nn.utils.prune as prune
    
    logger.info(f"Pruning model by {amount*100}%...")
    
    parameters_to_prune = []
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            parameters_to_prune.append((module, 'weight'))
    
    # Apply global pruning
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount
    )
    
    # Make pruning permanent
    for module, param_name in parameters_to_prune:
        prune.remove(module, param_name)
    
    # Count remaining parameters
    total_params = sum(p.numel() for p in model.parameters())
    nonzero_params = sum((p != 0).sum().item() for p in model.parameters())
    
    logger.info(f" Pruning complete")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Non-zero parameters: {nonzero_params:,}")
    logger.info(f"Sparsity: {((total_params - nonzero_params) / total_params * 100):.2f}%")
    
    return model


def quantize_model(model: nn.Module, calibration_data: torch.Tensor = None) -> nn.Module:
    """
    Quantize model to INT8 for faster inference
    
    Args:
        model: PyTorch model
        calibration_data: Optional calibration data for static quantization
    
    Returns:
        Quantized model
    """
    logger.info("Quantizing model to INT8...")
    
    model.eval()
    model.cpu()  # Quantization typically done on CPU
    
    # Dynamic quantization (easier, no calibration needed)
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Conv2d, nn.Linear},
        dtype=torch.qint8
    )
    
    logger.info(" Quantization complete")
    
    # Calculate model size reduction
    def get_model_size(model):
        torch.save(model.state_dict(), "temp_model.pth")
        size = Path("temp_model.pth").stat().st_size / (1024 * 1024)
        Path("temp_model.pth").unlink()
        return size
    
    original_size = get_model_size(model)
    quantized_size = get_model_size(quantized_model)
    
    logger.info(f"Original size: {original_size:.2f} MB")
    logger.info(f"Quantized size: {quantized_size:.2f} MB")
    logger.info(f"Size reduction: {((original_size - quantized_size) / original_size * 100):.2f}%")
    
    return quantized_model


def optimize_for_mobile(
    checkpoint_path: str,
    output_path: str,
    model_type: str = 'unet_light'
):
    """
    Optimize model for mobile deployment
    Applies pruning and quantization
    
    Args:
        checkpoint_path: Path to trained checkpoint
        output_path: Path to save optimized model
        model_type: Model architecture type
    """
    logger.info(f"Optimizing {model_type} for mobile deployment...")
    
    # Load model
    if model_type == 'unet_light':
        model = UNetLight()
    elif model_type == 'unet_standard':
        model = UNetStandard()
    elif model_type == 'unet_attention':
        model = UNetAttention()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Apply optimizations
    logger.info("\n" + "="*60)
    logger.info("Step 1: Pruning")
    logger.info("="*60)
    model = prune_model(model, amount=0.3)
    
    logger.info("\n" + "="*60)
    logger.info("Step 2: Quantization")
    logger.info("="*60)
    model = quantize_model(model)
    
    # Save optimized model
    torch.save(model.state_dict(), output_path)
    logger.info(f"\n Optimized model saved: {output_path}")


def benchmark_optimization(original_model, optimized_model, num_iterations: int = 100):
    """Benchmark speed and accuracy of optimizations"""
    import time
    
    logger.info(f"\nBenchmarking ({num_iterations} iterations)...")
    
    # Dummy input
    dummy_input = torch.randn(1, 3, 256, 256)
    
    # Benchmark original
    original_model.eval()
    start = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = original_model(dummy_input)
    original_time = time.time() - start
    
    # Benchmark optimized
    optimized_model.eval()
    start = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = optimized_model(dummy_input)
    optimized_time = time.time() - start
    
    logger.info(f"Original: {original_time/num_iterations*1000:.2f} ms/image")
    logger.info(f"Optimized: {optimized_time/num_iterations*1000:.2f} ms/image")
    logger.info(f"Speedup: {original_time/optimized_time:.2f}x")


def main():
    """Main optimization function"""
    parser = argparse.ArgumentParser(description="Optimize models for edge deployment")
    parser.add_argument('--checkpoint', type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument('--output', type=str, required=True,
                       help="Path to save optimized model")
    parser.add_argument('--model-type', type=str, default='unet_light',
                       choices=['unet_light', 'unet_standard', 'unet_attention'],
                       help="Model architecture")
    parser.add_argument('--prune-only', action='store_true',
                       help="Only apply pruning")
    parser.add_argument('--quantize-only', action='store_true',
                       help="Only apply quantization")
    
    args = parser.parse_args()
    
    optimize_for_mobile(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        model_type=args.model_type
    )


if __name__ == "__main__":
    main()
