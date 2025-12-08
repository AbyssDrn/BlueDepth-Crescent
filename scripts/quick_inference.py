#!/usr/bin/env python3
"""
BlueDepth-Crescent Quick Inference Script
Run inference on a single image or directory quickly
"""

import sys
import argparse
from pathlib import Path
import torch
from PIL import Image
import torchvision.transforms as transforms

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import UNetAttention
from utils.visualization import save_comparison_grid

def enhance_image(model, image_path: Path, output_path: Path, device: torch.device):
    """Enhance a single image"""
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Inference
    model.eval()
    with torch.no_grad():
        enhanced = model(input_tensor)
    
    # Save
    enhanced_image = transforms.ToPILImage()(enhanced.squeeze(0).cpu())
    enhanced_image.save(output_path)
    
    print(f" Saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Quick image enhancement")
    parser.add_argument('--input', required=True, help="Input image or directory")
    parser.add_argument('--output', default='results/enhanced', help="Output directory")
    parser.add_argument('--checkpoint', default='checkpoints/unet_attention_best.pth', help="Model checkpoint")
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = UNetAttention().to(device)
    if Path(args.checkpoint).exists():
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
        print(f" Loaded: {args.checkpoint}")
    else:
        print(f" Checkpoint not found: {args.checkpoint}")
        return 1
    
    # Create output dir
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single image
        output_path = output_dir / f"enhanced_{input_path.name}"
        enhance_image(model, input_path, output_path, device)
    elif input_path.is_dir():
        # Directory
        images = list(input_path.glob('*.jpg')) + list(input_path.glob('*.png'))
        print(f"Found {len(images)} images")
        
        for img_path in images:
            output_path = output_dir / f"enhanced_{img_path.name}"
            enhance_image(model, img_path, output_path, device)
    else:
        print(f" Invalid input: {input_path}")
        return 1
    
    print(f"\n Done! Results saved to {output_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
