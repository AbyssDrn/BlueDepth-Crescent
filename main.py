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
        print(f"[ERROR] Missing dependency: {e}")
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
        "unet_standard_best.pth",
        "unet_attention_best.pth",
        "unet_light_best.pth",
        "*_best.pth",
        "*_final.pth",
        "*_epoch_*.pth"
    ]
    
    for pattern in patterns:
        matches = sorted(checkpoint_dir.glob(pattern))
        if matches:
            return str(matches[-1])
    
    return None

def list_checkpoints(args):
    """List all available checkpoints with details"""
    checkpoint_dir = Path("checkpoints")
    
    if not checkpoint_dir.exists():
        print("[ERROR] Checkpoint directory not found!")
        print("\n[TIP] Create checkpoints by training a model:")
        print("   python main.py train --model standard --epochs 100")
        return
    
    checkpoints = sorted(checkpoint_dir.glob("*.pth"))
    
    if not checkpoints:
        print("[ERROR] No checkpoints found!")
        print("\n[TIP] Train a model first:")
        print("   python main.py train --model standard")
        return
    
    print("\n[INFO] Available Checkpoints:")
    print("=" * 80)
    
    for i, ckpt in enumerate(checkpoints, 1):
        size_mb = ckpt.stat().st_size / (1024 * 1024)
        
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
        print(f"[RECOMMENDED] Auto-detected: {Path(best).name}")

def train_model(args):
    """Train enhancement model"""
    from training.train_unet import UNetTrainer
    
    print(f"\n[TRAINING] Starting Training: {args.model.upper()}")
    print("=" * 70)
    
    # Check data directory
    data_base = Path(args.data_dir)
    train_hazy = data_base / 'train' / 'hazy'
    train_clear = data_base / 'train' / 'clear'
    val_hazy = data_base / 'val' / 'hazy'
    val_clear = data_base / 'val' / 'clear'
    
    # Alternative: test instead of val
    if not val_hazy.exists():
        val_hazy = data_base / 'test' / 'hazy'
        val_clear = data_base / 'test' / 'clear'
    
    # Check if directories exist
    if not train_hazy.exists():
        print(f"[ERROR] Data directories not found!")
        print(f"\n[STRUCTURE] Expected structure:")
        print(f"  {data_base}/")
        print(f"    train/")
        print(f"      hazy/     (training hazy images)")
        print(f"      clear/    (training clear/ground truth images)")
        print(f"    val/ or test/")
        print(f"      hazy/     (validation hazy images)")
        print(f"      clear/    (validation clear images)")
        print(f"\n[ACTION] Organize your data first:")
        print(f"  python organize_data.py")
        sys.exit(1)
    
    # Check for clear images
    has_clear_train = train_clear.exists() and len(list(train_clear.glob('*'))) > 0
    has_clear_val = val_clear.exists() and len(list(val_clear.glob('*'))) > 0
    
    if has_clear_train and has_clear_val:
        print(f"[INFO] Running in SUPERVISED mode (paired hazy/clear)")
    else:
        print(f"[WARNING] No clear images found! Training requires paired data.")
        sys.exit(1)
    
    # Initialize trainer - NO MODEL OBJECT, just pass model_type string
    print(f"\n[INITIALIZING] Creating trainer...")
    
    try:
        trainer = UNetTrainer(
            model_type=args.model,           # STRING: 'standard', 'light', or 'attention'
            data_dir=str(data_base),         # Base data directory
            img_size=256,                    # Image size
            batch_size=args.batch_size,      # Batch size
            num_epochs=args.epochs,          # Number of epochs
            learning_rate=args.lr,           # Learning rate
            checkpoint_dir='checkpoints',    # Checkpoint directory
            log_dir='logs',                  # Log directory
            use_amp=True,                    # Mixed precision
            use_perceptual_loss=False        # Perceptual loss
        )
    except Exception as e:
        print(f"\n[ERROR] Failed to initialize trainer: {e}")
        print(f"\n[DEBUG] Arguments:")
        print(f"  model_type: {args.model}")
        print(f"  data_dir: {data_base}")
        print(f"  batch_size: {args.batch_size}")
        print(f"  epochs: {args.epochs}")
        print(f"  learning_rate: {args.lr}")
        raise
    
    # Check if resuming from checkpoint
    if hasattr(args, 'resume') and args.resume:
        checkpoint_path = Path(args.resume)
        if checkpoint_path.exists():
            print(f"[INFO] Resuming from checkpoint: {checkpoint_path}")
            trainer.load_checkpoint(str(checkpoint_path))
        else:
            print(f"[WARNING] Checkpoint not found: {checkpoint_path}")
    
    # Train
    print(f"\n[TRAINING] Training for {args.epochs} epochs...")
    print(f"[INFO] Model: {args.model.upper()}")
    print(f"[INFO] Batch size: {args.batch_size}")
    print(f"[INFO] Learning rate: {args.lr}")
    print(f"[INFO] Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"[INFO] Mixed precision: {'Enabled' if torch.cuda.is_available() else 'Disabled'}")
    print("=" * 70)
    
    trainer.train()
    
    print("\n[SUCCESS] Training Complete!")
    print(f"  Best checkpoint: checkpoints/{args.model}_best.pth")
    print(f"  TensorBoard logs: logs/{args.model}/")
    print(f"\n[NEXT STEPS]")
    print(f"  View training logs: tensorboard --logdir logs")
    print(f"  Test model: python main.py enhance --input data/test/hazy/image_001.jpg --output enhanced.jpg")


def enhance_image(args):
    """Enhance single image"""
    from inference import ImageEnhancer
    from PIL import Image
    
    print(f"\n[ENHANCE] Enhancing Image: {args.input}")
    print("=" * 70)
    
    if not Path(args.input).exists():
        print(f"[ERROR] Input file not found: {args.input}")
        sys.exit(1)
    
    model_path = args.model_path
    if not Path(model_path).exists():
        print(f"[INFO] Checkpoint '{model_path}' not found. Searching...")
        model_path = find_best_checkpoint()
        
        if model_path is None:
            print("\n[ERROR] No checkpoint files found!")
            print("\n[OPTIONS]:")
            print("  1. Train a model: python main.py train")
            print("  2. Download pretrained: see checkpoints/README.md")
            sys.exit(1)
        
        print(f"[INFO] Using checkpoint: {model_path}")
    
    enhancer = ImageEnhancer(model_path, device=args.device)
    
    print(f"\n[PROCESSING] Enhancing...")
    image = Image.open(args.input)
    enhanced, metrics = enhancer.enhance(image, return_metrics=True)
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(enhanced).save(output_path)
    
    print(f"\n[SUCCESS] Enhancement Complete!")
    print(f"  Inference time: {metrics['time']:.3f}s")
    print(f"  FPS: {metrics['fps']:.1f}")
    print(f"  Output: {output_path}")

def enhance_batch(args):
    """Enhance batch of images"""
    from inference import BatchProcessor
    
    print(f"\n[BATCH] Batch Enhancement")
    print("=" * 70)
    
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"[ERROR] Input directory not found: {input_dir}")
        sys.exit(1)
    
    model_path = args.model_path
    if not Path(model_path).exists():
        print(f"[INFO] Checkpoint '{model_path}' not found. Searching...")
        model_path = find_best_checkpoint()
        
        if model_path is None:
            print("\n[ERROR] No checkpoint files found!")
            sys.exit(1)
        
        print(f"[INFO] Using checkpoint: {model_path}")
    
    processor = BatchProcessor(model_path, device=args.device)
    
    print(f"\n[PROCESSING] Processing images from: {input_dir}")
    stats = processor.process_directory(
        str(input_dir),
        args.output_dir,
        batch_size=args.batch_size
    )
    
    print(f"\n[SUCCESS] Batch Enhancement Complete!")
    print(f"  Processed: {stats['total_images']} images")
    print(f"  Total time: {stats['total_time']:.2f}s")
    print(f"  Average time: {stats['avg_time']:.3f}s per image")
    print(f"  Output: {args.output_dir}")

def process_video(args):
    """Process video file"""
    from inference import VideoProcessor
    
    print(f"\n[VIDEO] Video Processing: {args.input}")
    print("=" * 70)
    
    if not Path(args.input).exists():
        print(f"[ERROR] Video file not found: {args.input}")
        sys.exit(1)
    
    model_path = args.model_path
    if not Path(model_path).exists():
        print(f"[INFO] Checkpoint '{model_path}' not found. Searching...")
        model_path = find_best_checkpoint()
        
        if model_path is None:
            print("\n[ERROR] No checkpoint files found!")
            sys.exit(1)
        
        print(f"[INFO] Using checkpoint: {model_path}")
    
    processor = VideoProcessor(model_path, device=args.device)
    
    print(f"\n[PROCESSING] Processing video...")
    
    def progress_callback(frame, total):
        percent = (frame / total) * 100
        print(f"  Progress: {frame}/{total} frames ({percent:.1f}%)", end='\r')
    
    stats = processor.process_video(
        args.input,
        args.output,
        progress_callback=progress_callback
    )
    
    print(f"\n\n[SUCCESS] Video Processing Complete!")
    print(f"  Total frames: {stats['total_frames']}")
    print(f"  Processing time: {stats['total_time']:.2f}s")
    print(f"  Average FPS: {stats['avg_fps']:.1f}")
    print(f"  Output: {args.output}")

def export_model(args):
    """Export model to ONNX"""
    from edge.export_onnx import export_to_onnx
    
    print(f"\n[EXPORT] Exporting Model to ONNX")
    print("=" * 70)
    
    if not Path(args.model_path).exists():
        print(f"[INFO] Model checkpoint not found: {args.model_path}")
        
        model_path = find_best_checkpoint()
        if model_path:
            print(f"[INFO] Found alternative: {model_path}")
            args.model_path = model_path
        else:
            print("[ERROR] No checkpoints found!")
            sys.exit(1)
    
    print(f"\n[PROCESSING] Exporting...")
    print(f"  Input: {args.model_path}")
    print(f"  Output: {args.output}")
    
    export_to_onnx(
        args.model_path,
        args.output,
        input_shape=args.input_shape,
        opset_version=args.opset
    )
    
    print(f"\n[SUCCESS] Export Complete!")
    print(f"  ONNX model: {args.output}")
    print(f"  Input shape: {args.input_shape}")

def launch_dashboard(args):
    """Launch Gradio dashboard"""
    import subprocess
    
    print(f"\n[DASHBOARD] Launching BlueDepth-Crescent Dashboard")
    print("=" * 70)
    
    dashboard_path = Path(__file__).parent / "ui" / "app.py"
    
    if not dashboard_path.exists():
        print(f"[ERROR] Dashboard file not found: {dashboard_path}")
        print("\n[TIP] Make sure ui/app.py exists")
        sys.exit(1)
    
    print(f"\n[INFO] Dashboard will open at: http://localhost:{args.port}")
    print(f"[INFO] Press Ctrl+C to stop\n")
    
    try:
        subprocess.run([
            sys.executable,
            str(dashboard_path),
            "--port", str(args.port),
            "--share" if args.share else "--no-share"
        ])
    except KeyboardInterrupt:
        print("\n\n[INFO] Dashboard stopped.")
    except Exception as e:
        print(f"\n[ERROR] Error launching dashboard: {e}")
        print(f"\n[TIP] Try running directly:")
        print(f"  python {dashboard_path}")

def run_tests(args):
    """Run test suite"""
    import pytest
    
    print(f"\n[TESTING] Running Test Suite")
    print("=" * 70)
    
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
    
    print(f"\n[INFO] Running tests: {test_path}")
    
    exit_code = pytest.main([
        test_path,
        '-v',
        '--tb=short',
        '--color=yes'
    ])
    
    if exit_code == 0:
        print(f"\n[SUCCESS] All tests passed!")
    else:
        print(f"\n[ERROR] Some tests failed. Exit code: {exit_code}")
        sys.exit(exit_code)

def show_info(args):
    """Show system and model information"""
    print(f"\n[INFO] System Information")
    print("=" * 70)
    
    print(f"\n[PYTHON]")
    print(f"  Version: {sys.version.split()[0]}")
    print(f"  Executable: {sys.executable}")
    
    try:
        import torch
        print(f"\n[PYTORCH]")
        print(f"  Version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    except ImportError:
        print(f"\n[WARNING] PyTorch not installed")
    
    print(f"\n[PROJECT STRUCTURE]")
    dirs = ['models', 'inference', 'training', 'data', 'utils', 'ui', 'tests', 'checkpoints', 'logs']
    for dir_name in dirs:
        exists = "[OK]" if Path(dir_name).exists() else "[MISSING]"
        print(f"  {exists} {dir_name}/")
    
    print(f"\n[CHECKPOINTS]")
    ckpt_dir = Path('checkpoints')
    if ckpt_dir.exists():
        checkpoints = list(ckpt_dir.glob('*.pth'))
        if checkpoints:
            for ckpt in checkpoints[:5]:
                size = ckpt.stat().st_size / (1024 * 1024)
                print(f"  {ckpt.name} ({size:.1f} MB)")
            if len(checkpoints) > 5:
                print(f"  ... and {len(checkpoints) - 5} more")
        else:
            print(f"  No checkpoints found")
    else:
        print(f"  Checkpoints directory not found")

def main():
    """Main entry point"""
    
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
  
  # Batch enhancement
  python main.py batch --input-dir data/test/hazy --output-dir data/enhanced
  
  # Process video
  python main.py video --input data/videos/underwater.mp4 --output enhanced.mp4
  
  # Export to ONNX
  python main.py export --model-path checkpoints/unet_standard_best.pth --output model.onnx
  
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
    train_parser.add_argument('--resume', type=str, default=None, 
                            help='Resume from checkpoint path')  # ADD THIS
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
    
    args = parser.parse_args()
    
    if args.command is None:
        print_banner()
        parser.print_help()
        sys.exit(1)
    
    if args.command not in ['info', 'list']:
        print_banner()
    
    args.func(args)

if __name__ == "__main__":
    main()
