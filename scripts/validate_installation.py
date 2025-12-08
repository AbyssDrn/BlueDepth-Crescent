#!/usr/bin/env python3
"""
BlueDepth-Crescent Installation Validator
Validates complete project setup, dependencies, and configurations
"""

import sys
import os
from pathlib import Path
import importlib
import subprocess
from typing import Dict, List, Tuple

# ANSI color codes
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
    """Print success message"""
    print(f"{Colors.GREEN} {text}{Colors.END}")

def print_error(text: str):
    """Print error message"""
    print(f"{Colors.RED} {text}{Colors.END}")

def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.YELLOW} {text}{Colors.END}")

def check_python_version() -> bool:
    """Check if Python version is >= 3.8"""
    print_header("Checking Python Version")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print_success(f"Python {version.major}.{version.minor}.{version.micro} ")
        return True
    else:
        print_error(f"Python {version.major}.{version.minor}.{version.micro} (Requires >= 3.8)")
        return False

def check_required_packages() -> Tuple[bool, List[str]]:
    """Check if all required packages are installed"""
    print_header("Checking Required Packages")
    
    required_packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'cv2': 'OpenCV (cv2)',
        'PIL': 'Pillow',
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib',
        'tqdm': 'tqdm',
        'yaml': 'PyYAML',
        'psutil': 'psutil',
        'pynvml': 'nvidia-ml-py3'
    }
    
    missing = []
    installed = []
    
    for package, name in required_packages.items():
        try:
            if package == 'cv2':
                importlib.import_module('cv2')
            elif package == 'PIL':
                importlib.import_module('PIL')
            elif package == 'yaml':
                importlib.import_module('yaml')
            else:
                importlib.import_module(package)
            print_success(f"{name:20} installed")
            installed.append(package)
        except ImportError:
            print_error(f"{name:20} NOT installed")
            missing.append(package)
    
    return len(missing) == 0, missing

def check_cuda_availability() -> bool:
    """Check CUDA availability"""
    print_header("Checking CUDA/GPU")
    
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            print_success(f"CUDA {cuda_version} available")
            print_success(f"GPU: {device_name}")
            print_success(f"Number of GPUs: {device_count}")
            return True
        else:
            print_warning("CUDA not available - will use CPU")
            return False
    except Exception as e:
        print_error(f"Error checking CUDA: {e}")
        return False

def check_project_structure() -> bool:
    """Check if project structure is correct"""
    print_header("Checking Project Structure")
    
    required_dirs = [
        'models',
        'inference',
        'training',
        'data',
        'utils',
        'configs',
        'scripts',
        'checkpoints',
        'logs',
        'results'
    ]
    
    required_files = [
        'models/__init__.py',
        'models/base_model.py',
        'models/unet_standard.py',
        'models/unet_light.py',
        'models/unet_attention.py',
        'models/classifier.py',
        'inference/__init__.py',
        'inference/enhancer.py',
        'inference/classifier.py',
        'inference/batch_processor.py',
        'inference/video_processor.py',
        'training/__init__.py',
        'training/train_unet.py',
        'training/dataset.py',
        'training/losses.py',
        'training/device_manager.py',
        'utils/__init__.py',
        'utils/config.py',
        'utils/logger.py',
        'utils/metrics.py',
        'utils/visualization.py',
        'utils/image_utils.py',
        'utils/video_utils.py',
        'configs/default.yaml',
        'configs/jetson_nano.yaml',
        'configs/rtx4050.yaml'
    ]
    
    all_good = True
    
    # Check directories
    for dir_name in required_dirs:
        if Path(dir_name).exists():
            print_success(f"Directory: {dir_name:20}")
        else:
            print_error(f"Directory: {dir_name:20} MISSING")
            all_good = False
    
    # Check files
    for file_path in required_files:
        if Path(file_path).exists():
            print_success(f"File: {file_path}")
        else:
            print_error(f"File: {file_path} MISSING")
            all_good = False
    
    return all_good

def check_data_directory() -> bool:
    """Check data directory structure"""
    print_header("Checking Data Directory")
    
    data_dirs = [
        'data/train/hazy',
        'data/train/clear',
        'data/test/hazy',
        'data/test/clear'
    ]
    
    all_good = True
    
    for dir_path in data_dirs:
        path = Path(dir_path)
        if path.exists():
            count = len(list(path.glob('*.jpg'))) + len(list(path.glob('*.png')))
            if count > 0:
                print_success(f"{dir_path:30} ({count} images)")
            else:
                print_warning(f"{dir_path:30} (empty)")
        else:
            print_warning(f"{dir_path:30} MISSING")
            all_good = False
    
    return all_good

def check_configs() -> bool:
    """Check configuration files"""
    print_header("Checking Configuration Files")
    
    import yaml
    
    config_files = [
        'configs/default.yaml',
        'configs/jetson_nano.yaml',
        'configs/rtx4050.yaml'
    ]
    
    all_good = True
    
    for config_path in config_files:
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            print_success(f"{config_path:30} valid")
        except FileNotFoundError:
            print_error(f"{config_path:30} NOT FOUND")
            all_good = False
        except yaml.YAMLError as e:
            print_error(f"{config_path:30} INVALID YAML: {e}")
            all_good = False
    
    return all_good

def test_model_imports() -> bool:
    """Test if models can be imported"""
    print_header("Testing Model Imports")
    
    models_to_test = [
        ('models.unet_standard', 'UNetStandard'),
        ('models.unet_light', 'UNetLight'),
        ('models.unet_attention', 'UNetAttention'),
        ('models.classifier', 'UnderwaterClassifier')
    ]
    
    all_good = True
    
    for module_name, class_name in models_to_test:
        try:
            module = importlib.import_module(module_name)
            getattr(module, class_name)
            print_success(f"{class_name:30} imported")
        except Exception as e:
            print_error(f"{class_name:30} FAILED: {e}")
            all_good = False
    
    return all_good

def test_model_instantiation() -> bool:
    """Test if models can be instantiated"""
    print_header("Testing Model Instantiation")
    
    try:
        import torch
        from models import UNetStandard, UNetLight, UNetAttention
        
        models = {
            'UNetLight': UNetLight(),
            'UNetStandard': UNetStandard(),
            'UNetAttention': UNetAttention()
        }
        
        for name, model in models.items():
            # Test forward pass
            dummy_input = torch.randn(1, 3, 256, 256)
            output = model(dummy_input)
            print_success(f"{name:20} instantiated and working")
        
        return True
    except Exception as e:
        print_error(f"Model instantiation failed: {e}")
        return False

def check_gpu_memory() -> bool:
    """Check GPU memory availability"""
    print_header("Checking GPU Memory")
    
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                total_memory = props.total_memory / (1024**3)  # GB
                print_success(f"GPU {i}: {total_memory:.2f} GB total memory")
            return True
        else:
            print_warning("No GPU available")
            return False
    except Exception as e:
        print_error(f"Error checking GPU memory: {e}")
        return False

def generate_report(results: Dict[str, bool]):
    """Generate final validation report"""
    print_header("VALIDATION REPORT")
    
    total = len(results)
    passed = sum(results.values())
    
    for test_name, result in results.items():
        status = f"{Colors.GREEN}PASS{Colors.END}" if result else f"{Colors.RED}FAIL{Colors.END}"
        print(f"{test_name:40} {status}")
    
    print(f"\n{Colors.BOLD}Total: {passed}/{total} checks passed{Colors.END}\n")
    
    if passed == total:
        print_success(" All validation checks passed! System ready for training.")
        return True
    else:
        print_error(f" {total - passed} check(s) failed. Please fix issues before proceeding.")
        return False

def main():
    """Main validation function"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}")
    print("")
    print("     BlueDepth-Crescent Installation Validator v1.0            ")
    print("     Maritime Security & Reconnaissance System                  ")
    print("")
    print(f"{Colors.END}\n")
    
    results = {}
    
    # Run all checks
    results['Python Version'] = check_python_version()
    packages_ok, missing = check_required_packages()
    results['Required Packages'] = packages_ok
    results['CUDA/GPU'] = check_cuda_availability()
    results['Project Structure'] = check_project_structure()
    results['Data Directory'] = check_data_directory()
    results['Configuration Files'] = check_configs()
    results['Model Imports'] = test_model_imports()
    results['Model Instantiation'] = test_model_instantiation()
    results['GPU Memory'] = check_gpu_memory()
    
    # Generate report
    all_passed = generate_report(results)
    
    # Print missing packages if any
    if missing:
        print_header("Missing Packages")
        print("Install missing packages with:")
        print(f"\n{Colors.YELLOW}pip install {' '.join(missing)}{Colors.END}\n")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
