"""Quick Deployment Script for BlueDepth-Crescent
Automates model setup, emoji removal, and validation
"""

import os
import sys
import subprocess
from pathlib import Path

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def run_command(cmd, description):
    """Run shell command and handle errors"""
    print(f"\n[*] {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"[+] {description} completed successfully")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[!] {description} failed")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python 3.10.11 is being used"""
    version = sys.version_info
    if version.major == 3 and version.minor == 10:
        print(f"[+] Python {version.major}.{version.minor}.{version.micro} detected")
        return True
    else:
        print(f"[!] Python {version.major}.{version.minor}.{version.micro} detected")
        print("[!] Python 3.10.11 required")
        return False

def check_venv():
    """Check if virtual environment is activated"""
    if sys.prefix != sys.base_prefix:
        print("[+] Virtual environment is activated")
        return True
    else:
        print("[!] Virtual environment not activated")
        print("[!] Please activate venv first:")
        print("    Windows: .\\venv\\Scripts\\activate")
        print("    Linux: source venv/bin/activate")
        return False

def remove_emojis():
    """Remove all emojis from codebase"""
    print_header("Step 1: Removing Emojis from Codebase")
    
    if Path("remove_emojis.py").exists():
        return run_command("python remove_emojis.py .", "Emoji removal")
    else:
        print("[!] remove_emojis.py not found, skipping...")
        return True

def install_pytorch():
    """Install PyTorch with CUDA support"""
    print_header("Step 2: Installing PyTorch with CUDA 11.8")
    
    cmd = "pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118"
    return run_command(cmd, "PyTorch installation")

def verify_cuda():
    """Verify CUDA availability"""
    print_header("Step 3: Verifying CUDA")
    
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            print(f"[+] CUDA is available")
            print(f"[+] GPU: {gpu_name}")
            print(f"[+] CUDA Version: {cuda_version}")
            return True
        else:
            print("[!] CUDA is not available")
            print("[!] Check NVIDIA drivers and CUDA installation")
            return False
    except ImportError:
        print("[!] PyTorch not installed")
        return False

def install_requirements():
    """Install remaining dependencies"""
    print_header("Step 4: Installing Dependencies")
    
    return run_command("pip install -r requirements.txt", "Dependencies installation")

def validate_installation():
    """Run validation script"""
    print_header("Step 5: Validating Installation")
    
    if Path("scripts/validate_installation.py").exists():
        return run_command("python scripts/validate_installation.py", "Installation validation")
    else:
        print("[!] Validation script not found")
        return True

def create_directories():
    """Create necessary directories"""
    print_header("Creating Project Directories")
    
    directories = [
        "data/raw/images",
        "data/raw/videos",
        "data/processed/enhanced",
        "data/processed/frames",
        "checkpoints/unet",
        "checkpoints/classifier",
        "logs/tensorboard",
        "logs/training",
        "models/onnx",
        "models/tensorrt",
        "results/enhanced",
        "results/detected"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("[+] All directories created")
    return True

def main():
    """Main deployment function"""
    print_header("BlueDepth-Crescent Quick Deployment")
    print("Maritime Security & Reconnaissance System")
    
    # Pre-checks
    if not check_python_version():
        sys.exit(1)
    
    if not check_venv():
        sys.exit(1)
    
    # Deployment steps
    steps = [
        ("Emoji Removal", remove_emojis),
        ("Directory Creation", create_directories),
        ("PyTorch Installation", install_pytorch),
        ("CUDA Verification", verify_cuda),
        ("Dependencies Installation", install_requirements),
        ("Installation Validation", validate_installation)
    ]
    
    failed_steps = []
    
    for step_name, step_func in steps:
        if not step_func():
            failed_steps.append(step_name)
            response = input(f"\n[?] {step_name} failed. Continue anyway? (y/n): ")
            if response.lower() != 'y':
                print("\n[!] Deployment aborted")
                sys.exit(1)
    
    # Summary
    print_header("Deployment Summary")
    
    if failed_steps:
        print(f"[!] Completed with {len(failed_steps)} warning(s):")
        for step in failed_steps:
            print(f"    - {step}")
    else:
        print("[+] All steps completed successfully!")
    
    print("\n" + "="*60)
    print("Next Steps:")
    print("="*60)
    print("1. Add underwater images to: data/raw/images/")
    print("2. Prepare dataset: python scripts/setup_dataset.py")
    print("3. Train models: python training/train_unet.py")
    print("4. Launch dashboard: python ui/dashboard.py")
    print("="*60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[!] Deployment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n[!] Unexpected error: {e}")
        sys.exit(1)