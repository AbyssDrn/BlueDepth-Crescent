#!/usr/bin/env python3
"""
BlueDepth-Crescent Main Entry Point
Advanced command-line interface for underwater image enhancement

Author: BlueDepth-Crescent Team
Version: 1.0.0
License: MIT
"""

import argparse
import sys
from pathlib import Path
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def print_banner():
    """Print BlueDepth-Crescent banner"""
    banner = """

                 BlueDepth-Crescent v1.0.0                    
         Underwater Vision Intelligence System                
                                                              
  AI-Powered Image Enhancement & Object Classification        

    """
    print(banner)


def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import torch
        import torchvision
        import cv2
        import PIL
        import numpy
        return True
    except ImportError as e:
        print(f" Missing dependency: {e}")
        print("\nInstall dependencies:")
        print("  pip install -r requirements.txt")
        return False


def find_best_checkpoint():
    """Find the best available checkpoint automatically"""
    checkpoint_dir = Path("checkpoints")
    
    if not checkpoint_dir.exists():
        return None
    
    # Priority order for finding checkpoints
    patterns = [
        "unet_standard_best.pth",   # Best standard model
        "unet_attention_best.pth",  # Best attention model
        "unet_light_best.pth",      # Best lightweight model
        "*_best.pth",               # Any best checkpoint
        "*_final.pth",              # Final checkpoint
        "*_epoch_*.pth"             # Any epoch checkpoint
    ]
    
    for pattern in patterns:
        matches = sorted(checkpoint_dir.glob(pattern))
        if matches:
            return str(matches[-1])  # Return the latest match
    
    return None


def list_checkpoints(args):
    """List all available checkpoints with details"""
    checkpoint_dir = Path("checkpoints")
    
    if not checkpoint_dir.exists():
        print(" Checkpoint directory not found!")
        print("\n Create checkpoints by training a model:")
        print("   python main.py train --model standard --epochs 100")
        return
    
    checkpoints = sorted(checkpoint_dir.glob("*.pth"))
    
    if not checkpoints:
        print(" No checkpoints found!")
        print("\n Train a model first:")
        print("   python main.py train --model standard")
        return
    
    print("\n Available Checkpoints:")
    print("=" * 80)
    
    for i, ckpt in enumerate(checkpoints, 1):
        size_mb = ckpt.stat().st_size / (1024 * 1024)
        
        # Try to load checkpoint info
        try:
            checkpoint = torch.load(ckpt, map_location='cpu')
            epoch = checkpoint.get('epoch', 'N/A')
            psnr = checkpoint.get('best_psnr', 0.0)
            ssim = checkpoint.get('best_ssim', 0.0)
            
            print(f"\n{i}. {ckpt.name}")
            print(f"   Size: {size_mb:.2f} MB")
            print(f"   Epoch: {epoch}")
            print(f"   PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f}")
            print(f"   Path: {ckpt}")
        except:
            print(f"\n{i}. {ckpt.name}")
            print(f"   Size: {size_mb:.2f} MB")
            print(f"   Path: {ckpt}")
    
    print("\n" + "=" * 80)
    
    best = find_best_checkpoint()
    if best:
        print(f" Recommended (auto-detected): {Path(best).name}")


def train_model(args):
    """Train enhancement model"""
    from training.train_unet import UNetTrainer
    from models import UNetLight, UNetStandard, UNetAttention
    from data import UnderwaterDataset
    from torch.utils.data import DataLoader
    
    print(f"\n Starting Training: {args.model.upper()}")
    print("=" * 60)
    
    # Select model
    model_map = {
        'light': UNetLight,
        'standard': UNetStandard,
        'attention': UNetAttention
    }
    
    if args.model not in model_map:
        print(f" Invalid model: {args.model}")
        print(f"   Choose from: {list(model_map.keys())}")
        sys.exit(1)
    
    model = model_map[args.model]()
    
    # Check data directory
    train_dir = Path(args.data_dir) / 'train'
    val_dir = Path(args.data_dir) / 'val'
    
    if not train_dir.exists() or not val_dir.exists():
        print(f" Data directory not found: {args.data_dir}")
        print("\n Organize your data:")
        print("   python scripts/organize_data.py")
        sys.exit(1)
    
    # Create datasets
    print(f"\n Loading dataset from: {args.data_dir}")
    train_dataset = UnderwaterDataset(str(train_dir), augment=True)
    val_dataset = UnderwaterDataset(str(val_dir), augment=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Validation samples: {len(val_dataset)}")
    
    # Initialize trainer
    trainer = UNetTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Train
    print(f"\n⏳ Training for {args.epochs} epochs...")
    history = trainer.train()
    
    print("\n Training Complete!")
    print(f"   Best PSNR: {history.get('best_psnr', 0):.2f} dB")
    print(f"   Best Epoch: {history.get('best_epoch', 0)}")
    print(f"   Checkpoint: checkpoints/{args.model}_best.pth")


def enhance_image(args):
    """Enhance single image"""
    from inference import ImageEnhancer
    from PIL import Image
    
    print(f"\n  Enhancing Image: {args.input}")
    print("=" * 60)
    
    # Check input file
    if not Path(args.input).exists():
        print(f" Input file not found: {args.input}")
        sys.exit(1)
    
    # Auto-detect checkpoint if needed
    model_path = args.model_path
    if not Path(model_path).exists():
        print(f"  Checkpoint '{model_path}' not found. Searching...")
        model_path = find_best_checkpoint()
        
        if model_path is None:
            print("\n No checkpoint files found!")
            print("\n Options:")
            print("   1. Train a model: python main.py train")
            print("   2. Download pretrained: see checkpoints/DOWNLOAD_MODELS.md")
            sys.exit(1)
        
        print(f" Using checkpoint: {model_path}")
    
    # Create enhancer
    enhancer = ImageEnhancer(model_path, device=args.device)
    
    # Enhance
    print(f"\n⏳ Processing...")
    image = Image.open(args.input)
    enhanced, metrics = enhancer.enhance(image, return_metrics=True)
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(enhanced).save(output_path)
    
    # Print results
    print(f"\n Enhancement Complete!")
    print(f"   Inference time: {metrics['time']:.3f}s")
    print(f"   FPS: {metrics['fps']:.1f}")
    print(f"   Output: {output_path}")


def enhance_batch(args):
    """Enhance batch of images"""
    from inference import BatchProcessor
    
    print(f"\n Batch Enhancement")
    print("=" * 60)
    
    # Check directories
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f" Input directory not found: {input_dir}")
        sys.exit(1)
    
    # Auto-detect checkpoint
    model_path = args.model_path
    if not Path(model_path).exists():
        print(f"  Checkpoint '{model_path}' not found. Searching...")
        model_path = find_best_checkpoint()
        
        if model_path is None:
            print("\n No checkpoint files found!")
            sys.exit(1)
        
        print(f" Using checkpoint: {model_path}")
    
    # Process batch
    processor = BatchProcessor(model_path, device=args.device)
    
    print(f"\n⏳ Processing images from: {input_dir}")
    stats = processor.process_directory(
        str(input_dir),
        args.output_dir,
        batch_size=args.batch_size
    )
    
    print(f"\n Batch Enhancement Complete!")
    print(f"   Processed: {stats['total_images']} images")
    print(f"   Total time: {stats['total_time']:.2f}s")
    print(f"   Average time: {stats['avg_time']:.3f}s per image")
    print(f"   Output: {args.output_dir}")


def process_video(args):
    """Process video file"""
    from inference import VideoProcessor
    
    print(f"\n Video Processing: {args.input}")
    print("=" * 60)
    
    # Check input
    if not Path(args.input).exists():
        print(f" Video file not found: {args.input}")
        sys.exit(1)
    
    # Auto-detect checkpoint
    model_path = args.model_path
    if not Path(model_path).exists():
        print(f"  Checkpoint '{model_path}' not found. Searching...")
        model_path = find_best_checkpoint()
        
        if model_path is None:
            print("\n No checkpoint files found!")
            sys.exit(1)
        
        print(f" Using checkpoint: {model_path}")
    
    # Process video
    processor = VideoProcessor(model_path, device=args.device)
    
    print(f"\n⏳ Processing video...")
    
    def progress_callback(frame, total):
        percent = (frame / total) * 100
        print(f"   Progress: {frame}/{total} frames ({percent:.1f}%)", end='\r')
    
    stats = processor.process_video(
        args.input,
        args.output,
        progress_callback=progress_callback
    )
    
    print(f"\n\n Video Processing Complete!")
    print(f"   Total frames: {stats['total_frames']}")
    print(f"   Processing time: {stats['total_time']:.2f}s")
    print(f"   Average FPS: {stats['avg_fps']:.1f}")
    print(f"   Output: {args.output}")


def export_model(args):
    """Export model to ONNX"""
    from edge.export_onnx import export_to_onnx
    
    print(f"\n Exporting Model to ONNX")
    print("=" * 60)
    
    # Check checkpoint
    if not Path(args.model_path).exists():
        print(f"  Model checkpoint not found: {args.model_path}")
        
        model_path = find_best_checkpoint()
        if model_path:
            print(f" Found alternative: {model_path}")
            args.model_path = model_path
        else:
            print(" No checkpoints found!")
            sys.exit(1)
    
    # Export
    print(f"\n⏳ Exporting...")
    print(f"   Input: {args.model_path}")
    print(f"   Output: {args.output}")
    
    export_to_onnx(
        args.model_path,
        args.output,
        input_shape=args.input_shape,
        opset_version=args.opset
    )
    
    print(f"\n Export Complete!")
    print(f"   ONNX model: {args.output}")
    print(f"   Input shape: {args.input_shape}")


def launch_dashboard(args):
    """Launch Gradio dashboard"""
    import subprocess
    
    print(f"\n Launching BlueDepth-Crescent Dashboard")
    print("=" * 60)
    
    # Get dashboard path
    dashboard_path = Path(__file__).parent / "ui" / "app.py"
    
    if not dashboard_path.exists():
        print(f" Dashboard file not found: {dashboard_path}")
        print("\n Make sure ui/app.py exists")
        sys.exit(1)
    
    print(f"\n Dashboard will open at: http://localhost:{args.port}")
    print(f"⏹  Press Ctrl+C to stop\n")
    
    try:
        # Launch Gradio dashboard
        subprocess.run([
            sys.executable,
            str(dashboard_path),
            "--port", str(args.port),
            "--share" if args.share else "--no-share"
        ])
    except KeyboardInterrupt:
        print("\n\n Dashboard stopped.")
    except Exception as e:
        print(f"\n Error launching dashboard: {e}")
        print(f"\n Try running directly:")
        print(f"   python {dashboard_path}")


def run_tests(args):
    """Run test suite"""
    import pytest
    
    print(f"\n Running Test Suite")
    print("=" * 60)
    
    # Determine which tests to run
    if args.test_type == 'all':
        test_path = 'tests/'
    elif args.test_type == 'models':
        test_path = 'tests/test_models.py'
    elif args.test_type == 'inference':
        test_path = 'tests/test_inference.py'
    elif args.test_type == 'training':
        test_path = 'tests/test_training.py'
    elif args.test_type == 'utils':
        test_path = 'tests/test_utils.py'
    else:
        test_path = 'tests/'
    
    print(f"\n⏳ Running tests: {test_path}")
    
    # Run pytest
    exit_code = pytest.main([
        test_path,
        '-v',
        '--tb=short',
        '--color=yes'
    ])
    
    if exit_code == 0:
        print(f"\n All tests passed!")
    else:
        print(f"\n Some tests failed. Exit code: {exit_code}")
        sys.exit(exit_code)


def show_info(args):
    """Show system and model information"""
    print(f"\n System Information")
    print("=" * 60)
    
    # Python info
    print(f"\n Python:")
    print(f"   Version: {sys.version.split()[0]}")
    print(f"   Executable: {sys.executable}")
    
    # PyTorch info
    try:
        import torch
        print(f"\n PyTorch:")
        print(f"   Version: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    except ImportError:
        print(f"\n  PyTorch not installed")
    
    # Project structure
    print(f"\n Project Structure:")
    dirs = ['models', 'inference', 'training', 'data', 'utils', 'ui', 'tests', 'checkpoints', 'logs']
    for dir_name in dirs:
        exists = "" if Path(dir_name).exists() else ""
        print(f"   {exists} {dir_name}/")
    
    # Checkpoints
    print(f"\n Checkpoints:")
    ckpt_dir = Path('checkpoints')
    if ckpt_dir.exists():
        checkpoints = list(ckpt_dir.glob('*.pth'))
        if checkpoints:
            for ckpt in checkpoints[:5]:  # Show first 5
                size = ckpt.stat().st_size / (1024 * 1024)
                print(f"    {ckpt.name} ({size:.1f} MB)")
            if len(checkpoints) > 5:
                print(f"   ... and {len(checkpoints) - 5} more")
        else:
            print(f"   No checkpoints found")
    else:
        print(f"   Checkpoints directory not found")


def main():
    """Main entry point"""
    
    # Check dependencies first
    if not check_dependencies():
        sys.exit(1)
    
    parser = argparse.ArgumentParser(
        description="BlueDepth-Crescent: Underwater Image Enhancement System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show system info
  python main.py info
  
  # List checkpoints
  python main.py list
  
  # Train model
  python main.py train --model standard --epochs 100 --batch-size 16
  
  # Enhance image (auto-detects checkpoint)
  python main.py enhance --input image.jpg --output enhanced.jpg
  
  # Enhance with specific checkpoint
  python main.py enhance --input image.jpg --output enhanced.jpg \\
      --model-path checkpoints/unet_standard_best.pth
  
  # Batch enhancement
  python main.py batch --input-dir data/raw --output-dir data/enhanced
  
  # Process video
  python main.py video --input video.mp4 --output enhanced.mp4
  
  # Export to ONNX
  python main.py export --model-path checkpoints/unet_standard_best.pth \\
      --output model.onnx
  
  # Launch dashboard
  python main.py dashboard --port 7860
  
  # Run tests
  python main.py test --type all

For more help: python main.py <command> --help
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show system information')
    info_parser.set_defaults(func=show_info)
    
    # List command
    list_parser = subparsers.add_parser('list', help='List available checkpoints')
    list_parser.set_defaults(func=list_checkpoints)
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train enhancement model')
    train_parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
    train_parser.add_argument('--model', type=str, default='standard', 
                            choices=['light', 'standard', 'attention'],
                            help='Model architecture')
    train_parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    train_parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    train_parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    train_parser.set_defaults(func=train_model)
    
    # Enhance command
    enhance_parser = subparsers.add_parser('enhance', help='Enhance single image')
    enhance_parser.add_argument('--input', type=str, required=True, help='Input image path')
    enhance_parser.add_argument('--output', type=str, required=True, help='Output image path')
    enhance_parser.add_argument('--model-path', type=str, default='checkpoints/unet_standard_best.pth',
                               help='Model checkpoint path')
    enhance_parser.add_argument('--device', type=str, default='cuda', 
                               choices=['cuda', 'cpu'], help='Device')
    enhance_parser.set_defaults(func=enhance_image)
    
    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Enhance batch of images')
    batch_parser.add_argument('--input-dir', type=str, required=True, help='Input directory')
    batch_parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    batch_parser.add_argument('--model-path', type=str, default='checkpoints/unet_standard_best.pth',
                             help='Model checkpoint path')
    batch_parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    batch_parser.add_argument('--device', type=str, default='cuda',
                             choices=['cuda', 'cpu'], help='Device')
    batch_parser.set_defaults(func=enhance_batch)
    
    # Video command
    video_parser = subparsers.add_parser('video', help='Process video')
    video_parser.add_argument('--input', type=str, required=True, help='Input video path')
    video_parser.add_argument('--output', type=str, required=True, help='Output video path')
    video_parser.add_argument('--model-path', type=str, default='checkpoints/unet_standard_best.pth',
                             help='Model checkpoint path')
    video_parser.add_argument('--device', type=str, default='cuda',
                             choices=['cuda', 'cpu'], help='Device')
    video_parser.set_defaults(func=process_video)
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export model to ONNX')
    export_parser.add_argument('--model-path', type=str, required=True, help='Model checkpoint')
    export_parser.add_argument('--output', type=str, required=True, help='Output ONNX file')
    export_parser.add_argument('--input-shape', type=int, nargs=4, 
                              default=[1, 3, 256, 256],
                              help='Input shape (batch, channels, height, width)')
    export_parser.add_argument('--opset', type=int, default=11, help='ONNX opset version')
    export_parser.set_defaults(func=export_model)
    
    # Dashboard command
    dashboard_parser = subparsers.add_parser('dashboard', help='Launch UI dashboard')
    dashboard_parser.add_argument('--port', type=int, default=7860, help='Server port')
    dashboard_parser.add_argument('--share', action='store_true', help='Create public link')
    dashboard_parser.set_defaults(func=launch_dashboard)
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Run tests')
    test_parser.add_argument('--type', type=str, default='all',
                            choices=['all', 'models', 'inference', 'training', 'utils'],
                            help='Test type to run')
    test_parser.set_defaults(func=run_tests)
    
    # Parse and execute
    args = parser.parse_args()
    
    if args.command is None:
        print_banner()
        parser.print_help()
        sys.exit(1)
    
    # Print banner for main commands
    if args.command not in ['info', 'list']:
        print_banner()
    
    # Execute command
    args.func(args)


if __name__ == "__main__":
    main()
