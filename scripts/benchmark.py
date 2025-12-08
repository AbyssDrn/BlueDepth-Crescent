#!/usr/bin/env python3
"""
BlueDepth-Crescent Benchmark Script
Benchmarks all models on test set and generates comparison report
"""

import sys
import time
import torch
import yaml
from pathlib import Path
from tqdm import tqdm
import json
from typing import Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import UNetLight, UNetStandard, UNetAttention
from training.dataset import UnderwaterDataset
from utils.metrics import MetricsCalculator
from utils.logger import setup_logger
from utils.device_manager import DeviceManager

# ANSI colors
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

logger = setup_logger(__name__)

def print_header(text: str):
    """Print colored header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text:^70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}\n")

def load_model(model_type: str, checkpoint_path: Path, device: torch.device):
    """Load model with checkpoint"""
    print(f"Loading {model_type}...")
    
    if model_type == "unet_light":
        model = UNetLight()
    elif model_type == "unet_standard":
        model = UNetStandard()
    elif model_type == "unet_attention":
        model = UNetAttention()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"{Colors.GREEN} Loaded checkpoint: {checkpoint_path}{Colors.END}")
    else:
        print(f"{Colors.YELLOW} No checkpoint found, using random weights{Colors.END}")
    
    model = model.to(device)
    model.eval()
    return model

def benchmark_model(
    model: torch.nn.Module,
    model_name: str,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device
) -> Dict:
    """Benchmark a single model"""
    print_header(f"Benchmarking {model_name}")
    
    metrics_list = []
    inference_times = []
    
    model.eval()
    with torch.no_grad():
        for hazy, clear in tqdm(test_loader, desc=f"Testing {model_name}"):
            hazy = hazy.to(device)
            clear = clear.to(device)
            
            # Measure inference time
            start_time = time.time()
            enhanced = model(hazy)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            # Calculate metrics
            for i in range(enhanced.shape[0]):
                metrics = MetricsCalculator.compute_all(
                    enhanced[i:i+1],
                    clear[i:i+1]
                )
                metrics_list.append(metrics)
    
    # Aggregate results
    avg_metrics = {
        'psnr': sum(m['psnr'] for m in metrics_list) / len(metrics_list),
        'ssim': sum(m['ssim'] for m in metrics_list) / len(metrics_list),
        'mse': sum(m['mse'] for m in metrics_list) / len(metrics_list),
        'mae': sum(m['mae'] for m in metrics_list) / len(metrics_list),
    }
    
    avg_time = sum(inference_times) / len(inference_times)
    fps = 1.0 / avg_time
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    results = {
        'model_name': model_name,
        'metrics': avg_metrics,
        'inference_time_ms': avg_time * 1000,
        'fps': fps,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'num_test_images': len(metrics_list)
    }
    
    return results

def print_results_table(results_list: List[Dict]):
    """Print formatted results table"""
    print_header("BENCHMARK RESULTS")
    
    # Header
    print(f"{'Model':20} {'PSNR':>8} {'SSIM':>8} {'Time(ms)':>10} {'FPS':>8} {'Params(M)':>12}")
    print("=" * 76)
    
    # Results
    for result in results_list:
        model_name = result['model_name']
        psnr = result['metrics']['psnr']
        ssim = result['metrics']['ssim']
        time_ms = result['inference_time_ms']
        fps = result['fps']
        params_m = result['total_parameters'] / 1e6
        
        print(f"{model_name:20} {psnr:8.2f} {ssim:8.4f} {time_ms:10.2f} {fps:8.2f} {params_m:12.2f}")
    
    print("=" * 76)
    
    # Find best model
    best_psnr = max(results_list, key=lambda x: x['metrics']['psnr'])
    best_speed = min(results_list, key=lambda x: x['inference_time_ms'])
    smallest = min(results_list, key=lambda x: x['total_parameters'])
    
    print(f"\n{Colors.BOLD}Best Quality:{Colors.END} {best_psnr['model_name']} (PSNR: {best_psnr['metrics']['psnr']:.2f})")
    print(f"{Colors.BOLD}Fastest:{Colors.END} {best_speed['model_name']} ({best_speed['fps']:.2f} FPS)")
    print(f"{Colors.BOLD}Smallest:{Colors.END} {smallest['model_name']} ({smallest['total_parameters']/1e6:.2f}M params)")

def save_results(results_list: List[Dict], output_path: Path):
    """Save results to JSON"""
    with open(output_path, 'w') as f:
        json.dump(results_list, f, indent=2)
    print(f"\n{Colors.GREEN} Results saved to {output_path}{Colors.END}")

def main():
    """Main benchmark function"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}")
    print("")
    print("         BlueDepth-Crescent Benchmark Script v1.0              ")
    print("")
    print(f"{Colors.END}\n")
    
    # Setup device
    device_manager = DeviceManager()
    device = device_manager.device
    
    print(f"Using device: {device}\n")
    
    # Load test dataset
    print("Loading test dataset...")
    test_dataset = UnderwaterDataset(
        hazy_dir="data/test/hazy",
        clear_dir="data/test/clear",
        img_size=256,
        augment=False
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2
    )
    
    print(f"Test dataset: {len(test_dataset)} images\n")
    
    # Models to benchmark
    models_config = [
        {
            'type': 'unet_light',
            'checkpoint': 'checkpoints/unet_light_best.pth',
            'name': 'UNet Light'
        },
        {
            'type': 'unet_standard',
            'checkpoint': 'checkpoints/unet_standard_best.pth',
            'name': 'UNet Standard'
        },
        {
            'type': 'unet_attention',
            'checkpoint': 'checkpoints/unet_attention_best.pth',
            'name': 'UNet Attention'
        }
    ]
    
    # Benchmark all models
    results_list = []
    
    for model_config in models_config:
        try:
            model = load_model(
                model_config['type'],
                Path(model_config['checkpoint']),
                device
            )
            
            results = benchmark_model(
                model,
                model_config['name'],
                test_loader,
                device
            )
            
            results_list.append(results)
            
            # Free memory
            del model
            torch.cuda.empty_cache() if device.type == 'cuda' else None
            
        except Exception as e:
            logger.error(f"Error benchmarking {model_config['name']}: {e}")
            continue
    
    # Print results
    if results_list:
        print_results_table(results_list)
        
        # Save results
        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)
        save_results(results_list, output_dir / "benchmark_results.json")
    else:
        print(f"{Colors.RED}No models were successfully benchmarked!{Colors.END}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
