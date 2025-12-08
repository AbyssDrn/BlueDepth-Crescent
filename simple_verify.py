#!/usr/bin/env python3
"""
Simple project verification (no dependencies)
"""
from pathlib import Path

def simple_verify():
    print("=" * 80)
    print(" SIMPLE PROJECT VERIFICATION")
    print("=" * 80)
    print()
    
    current = Path.cwd()
    print(f" Current directory: {current}")
    print()
    
    # Check essential structure
    checks = {
        "data/hazy": "Training hazy images",
        "data/clear": "Training clear images", 
        "data/test/input": "Test images",
        "data/test/output": "Test output (auto-generated)",
        "models": "Model architectures",
        "training": "Training scripts",
        "inference": "Inference scripts",
        "utils": "Utility functions",
        "configs": "Configuration files",
        "main.py": "Main entry point",
        "requirements.txt": "Dependencies",
        "test_cuda.py": "CUDA test script"
    }
    
    print(" Checking project structure:")
    print()
    
    all_good = True
    for item, description in checks.items():
        path = Path(item)
        exists = path.exists()
        
        if exists:
            if path.is_dir():
                # Count files in directory
                if "data" in item:
                    files = list(path.glob("*.jpg")) + list(path.glob("*.png"))
                    count = len(files)
                    print(f"    {item}/ - {count} images")
                else:
                    print(f"    {item}/")
            else:
                print(f"    {item}")
        else:
            print(f"    {item} - MISSING")
            all_good = False
    
    print()
    print("=" * 80)
    
    if all_good:
        print(" ALL ESSENTIAL FILES FOUND!")
        print()
        print(" DATA SUMMARY:")
        
        # Count images
        hazy_count = len(list(Path("data/hazy").glob("*.jpg")) + list(Path("data/hazy").glob("*.png")))
        clear_count = len(list(Path("data/clear").glob("*.jpg")) + list(Path("data/clear").glob("*.png")))
        test_count = len(list(Path("data/test/input").glob("*.jpg")) + list(Path("data/test/input").glob("*.png")))
        
        print(f"   Training pairs: {hazy_count} hazy + {clear_count} clear")
        print(f"   Test images: {test_count}")
        print()
        
        if hazy_count == clear_count and hazy_count > 0:
            print(" Training data looks good!")
        else:
            print("  Training data mismatch!")
        
        if test_count > 0:
            print(" Test data ready!")
        else:
            print("  No test images found!")
        
        print()
        print("=" * 80)
        print(" NEXT STEPS:")
        print("=" * 80)
        print()
        print("1. Create virtual environment:")
        print("   python -m venv venv")
        print()
        print("2. Activate it:")
        print("   .\\venv\\Scripts\\Activate.ps1")
        print()
        print("3. Install PyTorch:")
        print("   pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118")
        print()
        print("4. Install other packages:")
        print("   pip install -r requirements.txt")
        print()
        print("5. Test CUDA:")
        print("   python test_cuda.py")
        print()
        print("6. Start training:")
        print("   python main.py train")
        
    else:
        print(" SOME FILES MISSING!")
        print()
        print("You may need to run the project generator scripts.")
    
    print("=" * 80)

if __name__ == "__main__":
    simple_verify()