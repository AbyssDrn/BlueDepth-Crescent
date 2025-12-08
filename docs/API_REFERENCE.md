# BlueDepth-Crescent API Reference

**Version:** 1.0.0  
**Last Updated:** December 8, 2025  
**Author:** BlueDepth-Crescent Team

---

## Table of Contents

1. [Overview](#overview)
2. [Models API](#models-api)
3. [Inference API](#inference-api)
4. [Training API](#training-api)
5. [Data API](#data-api)
6. [Utilities API](#utilities-api)
7. [Configuration API](#configuration-api)
8. [Edge Deployment API](#edge-deployment-api)
9. [Examples](#examples)

---

## Overview

BlueDepth-Crescent provides a comprehensive API for underwater image enhancement and object classification. This document covers all public APIs available for developers.

### Installation

pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt


### Quick Start

from models import UNetStandard
from inference import ImageEnhancer
from PIL import Image

Load model
model = UNetStandard()
model.load_checkpoint('checkpoints/unet_standard_best.pth')

Create enhancer
enhancer = ImageEnhancer(model)

Enhance image
image = Image.open('underwater.jpg')
enhanced = enhancer.enhance(image)
enhanced.save('enhanced.jpg')


---

## Models API

### Base Model

#### `BaseModel`

Base class for all enhancement models.

**Location:** `models/base_model.py`

class BaseModel(nn.Module):
"""Base model with common functionality"""

def __init__(self):
    """Initialize base model"""
    
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass - must be implemented by subclasses"""
    
def load_checkpoint(self, path: str, device: str = 'cuda') -> dict:
    """
    Load model checkpoint
    
    Args:
        path: Path to checkpoint file
        device: Device to load model on ('cuda' or 'cpu')
        
    Returns:
        dict: Checkpoint metadata
    """
    
def save_checkpoint(
    self, 
    path: str, 
    epoch: int,
    optimizer_state: dict = None,
    **kwargs
):
    """
    Save model checkpoint
    
    Args:
        path: Save path
        epoch: Current epoch
        optimizer_state: Optimizer state dict
        **kwargs: Additional metadata
    """


**Example:**

from models import UNetStandard

model = UNetStandard()
checkpoint = model.load_checkpoint('checkpoints/best.pth')

print(f"Loaded epoch: {checkpoint['epoch']}")
print(f"Best PSNR: {checkpoint['best_psnr']:.2f} dB")


---

### UNet Models

#### `UNetLight`

Lightweight U-Net for edge devices.

**Parameters:** ~350K  
**Location:** `models/unet_light.py`

class UNetLight(BaseModel):
"""
Lightweight U-Net architecture

Args:
    in_channels: Input channels (default: 3)
    out_channels: Output channels (default: 3)
    init_features: Initial feature channels (default: 32)
"""

def __init__(
    self, 
    in_channels: int = 3,
    out_channels: int = 3,
    init_features: int = 32
):
    super().__init__()


**Example:**

from models import UNetLight
import torch

model = UNetLight()
x = torch.randn(1, 3, 256, 256)
output = model(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Parameters: {model.count_parameters():,}")


---

#### `UNetStandard`

Standard U-Net with skip connections.

**Parameters:** ~7.8M  
**Location:** `models/unet_standard.py`

class UNetStandard(BaseModel):
"""
Standard U-Net architecture

Args:
    in_channels: Input channels (default: 3)
    out_channels: Output channels (default: 3)
    features: Feature channels per level (default: )[5][6]
"""

**Best for:** High-quality enhancement, research

---

#### `UNetAttention`

U-Net with attention mechanisms.

**Parameters:** ~8.2M  
**Location:** `models/unet_attention.py`

class UNetAttention(BaseModel):
"""
U-Net with attention gates

Args:
    in_channels: Input channels (default: 3)
    out_channels: Output channels (default: 3)
    features: Feature channels (default: )[6][5]
"""


**Best for:** Complex underwater scenes, best quality

---

### Classifier

#### `UnderwaterClassifier`

Object classification for underwater images.

**Location:** `models/classifier.py`

class UnderwaterClassifier(BaseModel):
"""
Underwater object classifier

Args:
    num_classes: Number of object classes
    backbone: Backbone architecture ('resnet50', 'efficientnet_b0')
    pretrained: Use pretrained weights
"""

def __init__(
    self,
    num_classes: int = 10,
    backbone: str = 'resnet50',
    pretrained: bool = True
):


**Example:**

from models import UnderwaterClassifier
import torch

classifier = UnderwaterClassifier(num_classes=10)
x = torch.randn(1, 3, 224, 224)
logits = classifier(x)

probs = torch.softmax(logits, dim=1)
predicted = torch.argmax(probs, dim=1)

print(f"Predictions: {predicted}")
print(f"Confidence: {probs.max().item():.4f}")


---

## Inference API

### ImageEnhancer

Single image enhancement.

**Location:** `inference/enhancer.py`

class ImageEnhancer:
"""
Image enhancement inference

Args:
    model: Enhancement model
    device: Compute device
    half_precision: Use FP16 (faster on GPU)
"""

def __init__(
    self,
    model: nn.Module,
    device: str = 'cuda',
    half_precision: bool = False
):
    
def enhance(
    self,
    image: Union[np.ndarray, Image.Image, str],
    return_metrics: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
    """
    Enhance single image
    
    Args:
        image: Input image (array, PIL Image, or path)
        return_metrics: Return performance metrics
        
    Returns:
        Enhanced image (and metrics if requested)
    """
    
def enhance_batch(
    self,
    images: List[Union[np.ndarray, Image.Image]],
    batch_size: int = 4
) -> List[np.ndarray]:
    """
    Enhance multiple images
    
    Args:
        images: List of images
        batch_size: Processing batch size
        
    Returns:
        List of enhanced images
    """

**Example:**

from inference import ImageEnhancer
from models import UNetStandard
from PIL import Image

Setup
model = UNetStandard()
model.load_checkpoint('checkpoints/unet_standard_best.pth')
enhancer = ImageEnhancer(model, device='cuda')

Single image
image = Image.open('input.jpg')
enhanced, metrics = enhancer.enhance(image, return_metrics=True)

print(f"Inference time: {metrics['time']:.3f}s")
print(f"FPS: {metrics['fps']:.1f}")

Batch processing
images = [Image.open(f'img_{i}.jpg') for i in range(10)]
enhanced_batch = enhancer.enhance_batch(images, batch_size=4)


---

### VideoProcessor

Video enhancement.

**Location:** `inference/video_processor.py`


---

### VideoProcessor

Video enhancement.

**Location:** `inference/video_processor.py`

class VideoProcessor:
"""
Video enhancement with temporal consistency

Args:
    model: Enhancement model
    device: Compute device
    temporal_window: Frames for temporal consistency
"""

def process_video(
    self,
    input_path: str,
    output_path: str,
    codec: str = 'mp4v',
    fps: Optional[int] = None,
    progress_callback: Optional[Callable] = None
) -> dict:
    """
    Process video file
    
    Args:
        input_path: Input video path
        output_path: Output video path
        codec: Video codec
        fps: Output FPS (None = same as input)
        progress_callback: Progress callback function
        
    Returns:
        Processing statistics dict
    """

**Example:**

from inference import VideoProcessor
from models import UNetLight

model = UNetLight()
model.load_checkpoint('checkpoints/unet_light_best.pth')

processor = VideoProcessor(model, device='cuda')

def progress(frame, total):
print(f"Processing: {frame}/{total} ({frame/total*100:.1f}%)")

stats = processor.process_video(
'underwater_video.mp4',
'enhanced_video.mp4',
progress_callback=progress
)

print(f"Total frames: {stats['total_frames']}")
print(f"Processing time: {stats['total_time']:.2f}s")
print(f"Average FPS: {stats['avg_fps']:.1f}")


---

### Classifier Inference

**Location:** `inference/classifier.py`

class UnderwaterClassifierInference:
"""
Object classification inference

Args:
    model: Classifier model
    class_names: List of class names
    device: Compute device
"""

def classify(
    self,
    image: Union[np.ndarray, Image.Image],
    top_k: int = 5,
    confidence_threshold: float = 0.5
) -> dict:
    """
    Classify objects in image
    
    Args:
        image: Input image
        top_k: Return top K predictions
        confidence_threshold: Minimum confidence
        
    Returns:
        Classification results dict
    """

**Example:**

from inference import UnderwaterClassifierInference
from models import UnderwaterClassifier

model = UnderwaterClassifier(num_classes=10)
model.load_checkpoint('checkpoints/classifier_best.pth')

classifier = UnderwaterClassifierInference(
model,
class_names=['fish', 'coral', 'debris', 'submarine', ...]
)

image = Image.open('underwater.jpg')
results = classifier.classify(image, top_k=5)

for det in results['detections']:
print(f"{det['class']}: {det['confidence']:.2%} ({det['threat']})")


---

## Training API

### Trainer

**Location:** `training/train_unet.py`

class UNetTrainer:
"""
U-Net training pipeline

Args:
    model: Model to train
    train_loader: Training data loader
    val_loader: Validation data loader
    config: Training configuration
"""

def train(self) -> dict:
    """
    Run training loop
    
    Returns:
        Training history dict
    """
    
def validate(self) -> dict:
    """
    Run validation
    
    Returns:
        Validation metrics
    """

**Example:**

from training import UNetTrainer
from models import UNetStandard
from data import UnderwaterDataset
from configs import get_training_config

Setup
config = get_training_config('unet_standard')
model = UNetStandard()

Data
train_dataset = UnderwaterDataset('data/train', augment=True)
val_dataset = UnderwaterDataset('data/val', augment=False)

Trainer
trainer = UNetTrainer(
model=model,
train_dataset=train_dataset,
val_dataset=val_dataset,
config=config
)

Train
history = trainer.train()

print(f"Best PSNR: {history['best_psnr']:.2f} dB")
print(f"Best epoch: {history['best_epoch']}")


---

## Data API

### Dataset

**Location:** `data/dataset.py`

class UnderwaterDataset(Dataset):
"""
Underwater image dataset

Args:
    data_dir: Dataset directory
    split: 'train', 'val', or 'test'
    transform: Image transforms
    augment: Enable augmentation
"""

def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get dataset item
    
    Returns:
        (hazy_image, clean_image) tuple
    """

**Example:**

from data import UnderwaterDataset
from torch.utils.data import DataLoader

dataset = UnderwaterDataset(
'data/train',
split='train',
augment=True
)

loader = DataLoader(
dataset,
batch_size=16,
shuffle=True,
num_workers=4
)

for hazy, clean in loader:
print(f"Hazy: {hazy.shape}, Clean: {clean.shape}")
break


---

## Utilities API

### Metrics

**Location:** `utils/metrics.py`

def calculate_psnr(
img1: Union[np.ndarray, torch.Tensor],
img2: Union[np.ndarray, torch.Tensor],
data_range: float = 255.0
) -> float:
"""Calculate PSNR"""

def calculate_ssim(
img1: Union[np.ndarray, torch.Tensor],
img2: Union[np.ndarray, torch.Tensor],
data_range: float = 255.0
) -> float:
"""Calculate SSIM"""

def calculate_uiqm(img: np.ndarray) -> float:
"""Calculate Underwater Image Quality Measure"""

class MetricsCalculator:
"""Comprehensive metrics calculator"""

@staticmethod
def compute_all(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    data_range: float = 255.0
) -> Dict[str, float]:
    """Compute all metrics"""


---

## Configuration API

### Config Manager

**Location:** `configs/__init__.py`

def get_training_config(model_name: str) -> dict:
"""Get training configuration"""

def get_inference_config() -> dict:
"""Get inference configuration"""

def load_config(path: str) -> dict:
"""Load configuration from file"""


---

## Edge Deployment API

### ONNX Export

**Location:** `edge/export.py`

def export_to_onnx(
model: nn.Module,
output_path: str,
input_shape: Tuple[int, ...] = (1, 3, 256, 256),
opset_version: int = 11
):
"""Export model to ONNX format"""

def validate_onnx_model(onnx_path: str) -> bool:
"""Validate ONNX model"""


---

## Examples

### Complete Enhancement Pipeline

from models import UNetStandard
from inference import ImageEnhancer
from utils.metrics import MetricsCalculator
from PIL import Image
import numpy as np

Load model
model = UNetStandard()
model.load_checkpoint('checkpoints/unet_standard_best.pth')

Create enhancer
enhancer = ImageEnhancer(model, device='cuda', half_precision=True)

Load image
original = np.array(Image.open('underwater.jpg'))

Enhance
enhanced = enhancer.enhance(original)

Calculate metrics
metrics = MetricsCalculator.compute_all(enhanced, original)
print(f"PSNR: {metrics['psnr']:.2f} dB")
print(f"SSIM: {metrics['ssim']:.4f}")
print(f"UIQM: {metrics['uiqm']:.4f}")

Save
Image.fromarray(enhanced).save('enhanced.jpg')


### Batch Processing with Progress

from pathlib import Path
from tqdm import tqdm

input_dir = Path('input_images')
output_dir = Path('output_images')
output_dir.mkdir(exist_ok=True)

images = list(input_dir.glob('*.jpg'))

for img_path in tqdm(images):
image = Image.open(img_path)
enhanced = enhancer.enhance(image)

output_path = output_dir / img_path.name
Image.fromarray(enhanced).save(output_path)


---

## Error Handling

All APIs raise appropriate exceptions:

try:
model.load_checkpoint('invalid_path.pth')
except FileNotFoundError:
print("Checkpoint not found")
except RuntimeError:
print("Invalid checkpoint format")


---

## Performance Tips

1. **Use GPU when available**

device = 'cuda' if torch.cuda.is_available() else 'cpu'


2. **Enable half precision for RTX GPUs**

enhancer = ImageEnhancer(model, half_precision=True)


3. **Batch processing is faster**

enhanced_batch = enhancer.enhance_batch(images, batch_size=8)

4. **Use multi-threading for I/O**

loader = DataLoader(dataset, num_workers=4)


---

## Support

- **Documentation:** `docs/`
- **Issues:** GitHub Issues
- **Email:** support@bluedepth-crescent.com

**Last Updated:** December 8, 2025
