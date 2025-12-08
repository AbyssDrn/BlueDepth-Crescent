#!/usr/bin/env python3
"""
BlueDepth-Crescent Pretrained Model Downloader
Downloads pretrained checkpoints for quick inference
"""

import sys
import os
import requests
from pathlib import Path
from tqdm import tqdm
import hashlib

# ANSI colors
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text: str):
    """Print colored header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text:^70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}\n")

def print_success(text: str):
    print(f"{Colors.GREEN} {text}{Colors.END}")

def print_error(text: str):
    print(f"{Colors.RED} {text}{Colors.END}")

def print_warning(text: str):
    print(f"{Colors.YELLOW} {text}{Colors.END}")

def download_file(url: str, destination: Path) -> bool:
    """Download file with progress bar"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        with open(destination, 'wb') as f, tqdm(
            desc=destination.name,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
        
        print_success(f"Downloaded: {destination}")
        return True
    except Exception as e:
        print_error(f"Download failed: {e}")
        return False

def verify_checksum(file_path: Path, expected_md5: str) -> bool:
    """Verify file checksum"""
    if not file_path.exists():
        return False
    
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    
    return md5_hash.hexdigest() == expected_md5

def main():
    """Main download function"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}")
    print("")
    print("      BlueDepth-Crescent Pretrained Model Downloader v1.0      ")
    print("")
    print(f"{Colors.END}\n")
    
    print_warning("Note: Pretrained models will be uploaded after initial training")
    print_warning("This script is a template for future model distribution")
    print()
    
    # Pretrained models configuration
    # TODO: Update URLs after training and hosting models
    models = {
        'unet_light': {
            'url': 'https://example.com/models/unet_light_best.pth',
            'md5': 'placeholder_md5_hash',
            'description': 'Lightweight model (350K params) - Fast inference'
        },
        'unet_standard': {
            'url': 'https://example.com/models/unet_standard_best.pth',
            'md5': 'placeholder_md5_hash',
            'description': 'Standard model (7.8M params) - Balanced'
        },
        'unet_attention': {
            'url': 'https://example.com/models/unet_attention_best.pth',
            'md5': 'placeholder_md5_hash',
            'description': 'Attention model (8.2M params) - Best quality'
        },
        'classifier': {
            'url': 'https://example.com/models/classifier_best.pth',
            'md5': 'placeholder_md5_hash',
            'description': 'Underwater type classifier'
        }
    }
    
    print_header("Available Pretrained Models")
    for name, info in models.items():
        print(f"{name:20} - {info['description']}")
    
    # Ask user which models to download
    print("\nWhich models would you like to download?")
    print("Enter model names separated by commas (or 'all' for all models):")
    user_input = input("> ").strip().lower()
    
    if user_input == 'all':
        models_to_download = list(models.keys())
    else:
        models_to_download = [m.strip() for m in user_input.split(',')]
    
    # Download selected models
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    for model_name in models_to_download:
        if model_name not in models:
            print_warning(f"Unknown model: {model_name}")
            continue
        
        model_info = models[model_name]
        filename = f"{model_name}_best.pth"
        destination = checkpoint_dir / filename
        
        print_header(f"Downloading {model_name}")
        
        # Check if already exists
        if destination.exists():
            print_warning(f"{filename} already exists")
            overwrite = input("Overwrite? (y/n): ").strip().lower()
            if overwrite != 'y':
                print("Skipped")
                continue
        
        # Download
        print(f"URL: {model_info['url']}")
        success = download_file(model_info['url'], destination)
        
        if success:
            # Verify checksum
            if verify_checksum(destination, model_info['md5']):
                print_success("Checksum verified")
            else:
                print_warning("Checksum mismatch (file may be corrupted)")
    
    print_header("DOWNLOAD COMPLETE")
    print_success("Pretrained models ready for use!")
    print(f"\n{Colors.BOLD}Next steps:{Colors.END}")
    print(f"  Run inference: python -m inference.enhancer --checkpoint checkpoints/unet_attention_best.pth\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
