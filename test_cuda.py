#!/usr/bin/env python3
"""Test CUDA installation and GPU availability"""
import torch
import sys

def test_cuda():
    print("="*60)
    print(" CUDA Installation Test")
    print("="*60)
    
    print(f"\nPython version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"\n CUDA is available!")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"  Compute Capability: {props.major}.{props.minor}")
            print(f"  Total Memory: {props.total_memory / 1024**3:.2f} GB")
            print(f"  Multi Processors: {props.multi_processor_count}")
        
        # Test tensor operations
        print("\n Testing tensor operations on GPU...")
        try:
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            z = x @ y
            print(" GPU tensor operations working!")
        except Exception as e:
            print(f" GPU tensor operations failed: {e}")
    else:
        print("\n  CUDA is NOT available!")
        print("Running on CPU only.")
        print("\nPossible reasons:")
        print("1. No NVIDIA GPU detected")
        print("2. CUDA not installed")
        print("3. PyTorch installed without CUDA support")
        print("\nInstall PyTorch with CUDA:")
        print("pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    test_cuda()
