# BlueDepth-Crescent

**Deep Learning-Powered Underwater Image Enhancement and Object Intelligence System**

![Python 3.10](https://img.shields.io/badge/python-3.10.11-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-orange.svg)
![CUDA](https://img.shields.io/badge/CUDA-11.8+-green.svg)
![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Model Architectures](#model-architectures)
- [Workflow](#workflow)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Edge Deployment](#edge-deployment)
- [Performance](#performance)
- [Testing](#testing)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Citation](#citation)
- [Contact](#contact)

---

## Overview

**BlueDepth-Crescent** is a production-grade underwater vision intelligence system designed for maritime security and reconnaissance operations. The system leverages state-of-the-art deep learning techniques to enhance underwater imagery degraded by light absorption, scattering, and color distortion.

### Core Capabilities

- **Image Enhancement**: Removes haze, corrects color casts, and restores visual clarity in underwater imagery
- **Object Classification**: Identifies and classifies underwater objects with confidence scoring
- **Real-time Processing**: GPU-accelerated inference with thermal safety monitoring
- **Edge Deployment**: Optimized for NVIDIA Jetson devices with TensorRT acceleration
- **Video Processing**: Frame extraction, quality assessment, and batch enhancement
- **Interactive Dashboard**: User-friendly Gradio interface for model interaction

### Technical Approach

BlueDepth-Crescent employs U-Net based encoder-decoder architectures with skip connections to preserve spatial information during the enhancement process. The system uses a combination of perceptual losses (SSIM, MS-SSIM) and pixel-wise losses (L1, L2) to maintain structural similarity while improving visual quality.

---

## Key Features

### Image Enhancement

- **Multiple U-Net Variants**: Standard (31M params), Light (8M params), Attention (42M params)
- **Color Restoration**: Red wavelength recovery, blue/green cast removal
- **Quality Metrics**: PSNR, SSIM, MSE, color difference evaluation
- **Adaptive Processing**: Automatic exposure and contrast adjustment
- **Format Support**: PNG, JPG, JPEG, BMP, TIFF, RAW

### GPU Safety & Optimization

- **Thermal Monitoring**: Real-time temperature tracking via NVML
- **Auto-Throttling**: Dynamic batch size adjustment on thermal events
- **Mixed Precision Training**: FP16 automatic mixed precision (AMP)
- **Memory Management**: Gradient accumulation for large models
- **Fallback Mechanisms**: Safe CPU mode with user permission

### Edge Deployment

- **ONNX Export**: Model serialization for cross-platform compatibility
- **TensorRT Optimization**: INT8/FP16 quantization for inference acceleration
- **Jetson Support**: Nano, Xavier NX, Xavier, Orin series
- **Real-time Inference**: 30+ FPS on Jetson Xavier with FP16

### Video Processing

- **Frame Extraction**: FFmpeg-based video decoding
- **Quality Assessment**: Sharpness, contrast, brightness scoring
- **Best Frame Selection**: Automatic selection of highest quality frames
- **Batch Processing**: Parallel processing with progress tracking

---

## System Architecture

### High-Level Architecture

```
Input Layer (Raw Underwater Images)
          |
          v
   Preprocessing Module
    - Normalization
    - Resizing
    - Augmentation
          |
          v
   Enhancement Pipeline
    - U-Net Encoder (Downsampling)
    - Bottleneck (Feature Extraction)
    - U-Net Decoder (Upsampling)
    - Skip Connections
          |
          v
   Post-Processing Module
    - Color Correction
    - Contrast Enhancement
    - Sharpening
          |
          v
   Output Layer (Enhanced Images)
```

### Component Architecture

```
BlueDepth-Crescent/
|
|-- models/                      # Neural Network Architectures
|   |-- base_model.py           # Abstract base class with common methods
|   |-- unet_standard.py        # Standard U-Net (31M params, best quality)
|   |-- unet_light.py           # Lightweight U-Net (8M params, edge deployment)
|   |-- unet_attention.py       # Attention U-Net (42M params, complex scenes)
|   |-- classifier.py           # Object classification head
|
|-- training/                    # Training Infrastructure
|   |-- train_unet.py           # Main training loop with validation
|   |-- losses.py               # Custom loss functions (SSIM, MS-SSIM, Perceptual)
|   |-- dataset.py              # PyTorch Dataset with augmentation
|   |-- device_manager.py       # GPU thermal monitoring and safety
|
|-- inference/                   # Inference Pipeline
|   |-- enhancer.py             # Single/batch image enhancement
|   |-- batch_processor.py      # Parallel batch processing
|   |-- video_processor.py      # Video frame extraction and enhancement
|   |-- classifier.py           # Object classification inference
|   |-- model_loader.py         # Checkpoint loading and model initialization
|   |-- inference_handler.py    # API-style inference wrapper
|
|-- utils/                       # Utility Functions
|   |-- config.py               # Configuration management
|   |-- logger.py               # Structured logging
|   |-- metrics.py              # PSNR, SSIM, MSE calculations
|   |-- visualization.py        # Plotting and comparison charts
|   |-- image_utils.py          # Image I/O and preprocessing
|   |-- video_utils.py          # Video manipulation utilities
|
|-- ui/                          # User Interface
|   |-- app.py                  # Gradio dashboard application
|
|-- edge/                        # Edge Deployment
|   |-- export_onnx.py          # PyTorch to ONNX conversion
|   |-- convert_trt.py          # ONNX to TensorRT optimization
|   |-- jetson_inference.py     # Jetson-optimized inference engine
|   |-- optimize_model.py       # Model quantization and pruning
|
|-- configs/                     # Configuration Files
|   |-- training_config.yaml    # Training hyperparameters
|   |-- device_config.yaml      # GPU settings and thresholds
|   |-- model_configs/          # Model-specific configurations
|
|-- tests/                       # Unit Tests
|   |-- test_models.py          # Model architecture tests
|   |-- test_inference.py       # Inference pipeline tests
|   |-- test_training.py        # Training loop tests
|   |-- test_utils.py           # Utility function tests
|   |-- conftest.py             # Pytest fixtures
```

---

## Model Architectures

### 1. U-Net Standard (31M Parameters)

**Architecture Overview:**

```
Input (3 x H x W)
    |
Encoder:
    Conv2D(3, 64) + BatchNorm + ReLU
    MaxPool(2x2)
    Conv2D(64, 128) + BatchNorm + ReLU
    MaxPool(2x2)
    Conv2D(128, 256) + BatchNorm + ReLU
    MaxPool(2x2)
    Conv2D(256, 512) + BatchNorm + ReLU
    MaxPool(2x2)
    |
Bottleneck:
    Conv2D(512, 1024) + BatchNorm + ReLU
    |
Decoder:
    UpConv(1024, 512) + Skip Connection + Conv2D
    UpConv(512, 256) + Skip Connection + Conv2D
    UpConv(256, 128) + Skip Connection + Conv2D
    UpConv(128, 64) + Skip Connection + Conv2D
    |
Output:
    Conv2D(64, 3) + Sigmoid
    |
Output (3 x H x W)
```

**Key Characteristics:**

- Best quality enhancement
- 5 encoder/decoder levels
- Skip connections at each level
- Batch normalization for stable training
- Suitable for RTX 4050 training
- ~45 FPS inference on RTX 4050

**Use Cases:**

- High-quality enhancement for archival
- Training on desktop GPUs
- Research and experimentation

### 2. U-Net Light (8M Parameters)

**Architecture Overview:**

```
Input (3 x H x W)
    |
Encoder:
    Conv2D(3, 32) + BatchNorm + ReLU
    MaxPool(2x2)
    Conv2D(32, 64) + BatchNorm + ReLU
    MaxPool(2x2)
    Conv2D(64, 128) + BatchNorm + ReLU
    MaxPool(2x2)
    |
Bottleneck:
    Conv2D(128, 256) + BatchNorm + ReLU
    |
Decoder:
    UpConv(256, 128) + Skip Connection + Conv2D
    UpConv(128, 64) + Skip Connection + Conv2D
    UpConv(64, 32) + Skip Connection + Conv2D
    |
Output:
    Conv2D(32, 3) + Sigmoid
    |
Output (3 x H x W)
```

**Key Characteristics:**

- Reduced channel count (32, 64, 128, 256)
- 4 encoder/decoder levels
- Optimized for edge devices
- Lower memory footprint
- ~120 FPS inference on RTX 4050
- ~35 FPS on Jetson Xavier NX

**Use Cases:**

- Edge deployment (Jetson Nano/Xavier)
- Real-time video processing
- Resource-constrained environments

### 3. U-Net Attention (42M Parameters)

**Architecture Overview:**

```
Input (3 x H x W)
    |
Encoder:
    Conv2D(3, 64) + BatchNorm + ReLU
    MaxPool(2x2)
    Conv2D(64, 128) + BatchNorm + ReLU
    MaxPool(2x2)
    Conv2D(128, 256) + BatchNorm + ReLU
    MaxPool(2x2)
    Conv2D(256, 512) + BatchNorm + ReLU
    MaxPool(2x2)
    |
Bottleneck:
    Conv2D(512, 1024) + BatchNorm + ReLU
    |
Decoder (with Attention Gates):
    UpConv(1024, 512) + Attention Gate + Skip Connection
    UpConv(512, 256) + Attention Gate + Skip Connection
    UpConv(256, 128) + Attention Gate + Skip Connection
    UpConv(128, 64) + Attention Gate + Skip Connection
    |
Output:
    Conv2D(64, 3) + Sigmoid
    |
Output (3 x H x W)
```

**Attention Gate Mechanism:**

```
Gate(x_skip, x_up):
    g = Conv2D(x_up, channels)
    x = Conv2D(x_skip, channels)
    psi = ReLU(g + x)
    psi = Conv2D(psi, 1) + Sigmoid
    return x_skip * psi
```

**Key Characteristics:**

- Attention gates at each skip connection
- Focuses on relevant regions
- Better for complex underwater scenes
- Higher computational cost
- ~30 FPS inference on RTX 4050

**Use Cases:**

- Complex underwater environments
- Scenes with multiple objects
- Maximum quality requirements

---

## Workflow

### Training Workflow

```
Step 1: Data Preparation
    - Organize images into data/train and data/val
    - Apply data augmentation (rotation, flip, color jitter)
    - Normalize to [0, 1] range
    |
    v
Step 2: Model Initialization
    - Load model architecture (Standard/Light/Attention)
    - Initialize weights (Kaiming initialization)
    - Setup optimizer (Adam with learning rate 1e-4)
    |
    v
Step 3: Training Loop
    For each epoch:
        For each batch:
            - Forward pass through U-Net
            - Compute loss (SSIM + L1)
            - Backward pass and gradient update
            - Update learning rate (CosineAnnealing)
        |
        - Validate on validation set
        - Compute metrics (PSNR, SSIM)
        - Save checkpoint if best model
        - Log to TensorBoard
    |
    v
Step 4: Model Selection
    - Load best checkpoint (highest PSNR)
    - Evaluate on test set
    - Export for deployment
```

### Inference Workflow

```
Step 1: Image Input
    - Load image from file/camera/video
    - Validate format and dimensions
    |
    v
Step 2: Preprocessing
    - Resize to model input size (256x256 or 512x512)
    - Normalize to [0, 1]
    - Convert to tensor (C x H x W)
    - Add batch dimension (1 x C x H x W)
    |
    v
Step 3: Model Inference
    - Load checkpoint (model.pth)
    - Set model to eval mode
    - Disable gradient computation
    - Forward pass through U-Net
    |
    v
Step 4: Post-processing
    - Remove batch dimension
    - Denormalize from [0, 1] to [0, 255]
    - Convert to numpy array
    - Resize to original dimensions
    |
    v
Step 5: Output
    - Save enhanced image
    - Compute quality metrics
    - Return results
```

### Video Processing Workflow

```
Step 1: Video Input
    - Load video file (MP4, AVI, MOV)
    - Extract metadata (FPS, resolution, duration)
    |
    v
Step 2: Frame Extraction
    - Use FFmpeg to extract frames
    - Save frames to temporary directory
    - Track frame indices
    |
    v
Step 3: Quality Assessment
    For each frame:
        - Compute sharpness (Laplacian variance)
        - Compute contrast (standard deviation)
        - Compute brightness (mean intensity)
        - Assign quality score
    |
    v
Step 4: Frame Selection
    - Rank frames by quality score
    - Select top N frames or best single frame
    |
    v
Step 5: Enhancement
    - Load enhancer model
    - Process selected frames
    - Save enhanced frames
    |
    v
Step 6: Video Reconstruction (Optional)
    - Use FFmpeg to create video from enhanced frames
    - Maintain original FPS and resolution
    - Add audio if present in original
```

### Edge Deployment Workflow

```
Step 1: Model Training (Desktop GPU)
    - Train on RTX 4050
    - Achieve target PSNR/SSIM
    - Save checkpoint (.pth)
    |
    v
Step 2: ONNX Export
    - Load PyTorch checkpoint
    - Create dummy input (1 x 3 x 256 x 256)
    - Export to ONNX (opset 11)
    - Validate ONNX model
    |
    v
Step 3: TensorRT Conversion (on Jetson)
    - Load ONNX model
    - Build TensorRT engine
    - Choose precision (FP32/FP16/INT8)
    - Optimize for Jetson architecture
    - Save TensorRT engine (.trt)
    |
    v
Step 4: Jetson Inference
    - Load TensorRT engine
    - Allocate CUDA buffers
    - Copy input to device
    - Execute inference
    - Copy output from device
    - Post-process and return
```

---

## Requirements

### Hardware Requirements

**Development Environment:**

- **GPU**: NVIDIA RTX 4050 (6GB VRAM) or better
- **RAM**: 16GB minimum (32GB recommended)
- **Storage**: 50GB for datasets, 10GB for models
- **CPU**: Intel i5 or AMD Ryzen 5 (4+ cores)

**Edge Deployment:**

- **Jetson Nano**: 4GB RAM, 128 CUDA cores
- **Jetson Xavier NX**: 8GB RAM, 384 CUDA cores
- **Jetson Xavier**: 32GB RAM, 512 CUDA cores
- **Jetson Orin**: 32GB RAM, 1024 CUDA cores (recommended)

### Software Requirements

**Operating System:**

- Ubuntu 20.04/22.04 (recommended)
- Windows 10/11
- macOS 12+ (CPU only)

**Core Dependencies:**

- Python 3.10.11
- CUDA 11.8+
- cuDNN 8.6+
- PyTorch 2.1.0
- TorchVision 0.16.0

Full dependency list in `requirements.txt`

---

## Installation

### Quick Installation

```bash
# Clone repository
git clone https://github.com/AbyssDrn/BlueDepth-Crescent.git
cd BlueDepth-Crescent

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PyTorch with CUDA 11.8
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu118

# Install remaining dependencies
pip install -r requirements.txt

# Verify installation
python scripts/verify_installation.py
```

### Automated Installation

**Linux/Mac:**

```bash
bash scripts/install.sh
```

**Windows:**

```bash
scripts\install.bat
```

### Verify CUDA

```bash
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

---

## Usage

### Command-Line Interface

BlueDepth-Crescent provides a unified CLI through `main.py`:

```bash
# Show system information
python main.py info

# List available checkpoints
python main.py list

# Train model
python main.py train --model standard --epochs 100

# Enhance single image
python main.py enhance --input underwater.jpg --output enhanced.jpg

# Batch enhancement
python main.py batch --input-dir data/raw --output-dir data/enhanced

# Process video
python main.py video --input video.mp4 --output enhanced_video.mp4

# Export to ONNX
python main.py export --model-path checkpoints/unet_standard_best.pth \
    --output model.onnx

# Launch dashboard
python main.py dashboard --port 7860

# Run tests
python main.py test --type all
```

### Training

**Basic Training:**

```bash
python main.py train --model standard --data-dir data --epochs 100
```

**Advanced Training:**

```bash
python main.py train \
    --model attention \
    --data-dir data \
    --batch-size 16 \
    --epochs 200 \
    --lr 0.0001
```

**Monitor Training:**

```bash
# TensorBoard
tensorboard --logdir=logs --port=6006

# Open browser to http://localhost:6006
```

### Inference

**Python API:**

```python
from inference import ImageEnhancer

# Load model
enhancer = ImageEnhancer('checkpoints/unet_standard_best.pth')

# Enhance single image
enhanced, metrics = enhancer.enhance('underwater.jpg', return_metrics=True)
print(f"PSNR: {metrics['psnr']:.2f} dB")

# Batch processing
enhancer.enhance_batch('data/raw', 'data/enhanced')
```

**Command Line:**

```bash
# Single image
python main.py enhance --input test.jpg --output result.jpg

# Batch
python main.py batch --input-dir images/ --output-dir enhanced/
```

### Video Processing

```python
from inference import VideoProcessor

processor = VideoProcessor('checkpoints/unet_standard_best.pth')

# Extract and enhance best frame
best_frame = processor.extract_best_frame('video.mp4')
enhanced = processor.enhance(best_frame)

# Process entire video
processor.process_video('input.mp4', 'output.mp4')
```

---

## Project Structure

```
BlueDepth-Crescent/
 models/                  # Neural network architectures
    __init__.py
    base_model.py       # Abstract base class
    unet_standard.py    # Standard U-Net (31M params)
    unet_light.py       # Lightweight U-Net (8M params)
    unet_attention.py   # Attention U-Net (42M params)
    classifier.py       # Classification head

 training/               # Training infrastructure
    __init__.py
    train_unet.py      # Main training script
    losses.py          # Loss functions (SSIM, MS-SSIM, Perceptual)
    dataset.py         # PyTorch Dataset with augmentation
    device_manager.py  # GPU thermal monitoring

 inference/              # Inference pipeline
    __init__.py
    enhancer.py        # Image enhancement
    batch_processor.py # Parallel batch processing
    video_processor.py # Video frame extraction
    classifier.py      # Object classification
    model_loader.py    # Model loading utilities
    inference_handler.py # API wrapper

 utils/                  # Utility functions
    __init__.py
    config.py          # Configuration management
    logger.py          # Structured logging
    metrics.py         # PSNR, SSIM, MSE
    visualization.py   # Plotting utilities
    image_utils.py     # Image I/O
    video_utils.py     # Video utilities

 ui/                     # User interface
    app.py             # Gradio dashboard
    assets/            # UI resources

 edge/                   # Edge deployment
    export_onnx.py     # ONNX export
    convert_trt.py     # TensorRT conversion
    jetson_inference.py # Jetson inference
    optimize_model.py  # Quantization

 configs/                # Configuration files
    training_config.yaml
    device_config.yaml
    model_configs/

 tests/                  # Unit tests
    test_models.py
    test_inference.py
    test_training.py
    test_utils.py
    conftest.py

 scripts/                # Automation scripts
    organize_data.py
    verify_installation.py
    install.sh
    install.bat

 docs/                   # Documentation
    ARCHITECTURE.md
    TRAINING_GUIDE.md
    DEPLOYMENT_GUIDE.md
    API_REFERENCE.md

 data/                   # Datasets (not in git)
    train/
    val/
    test/
    README.md

 checkpoints/            # Model checkpoints (not in git)
    unet_standard_best.pth
    unet_light_best.pth
    README.md

 logs/                   # Training logs (not in git)
    README.md

 results/                # Output results (not in git)
    enhanced/
    benchmarks/
    README.md

 main.py                 # Command-line interface
 remove_emojis.py       # Repository cleaner
 requirements.txt       # Python dependencies
 .gitignore            # Git exclusions
 .gitattributes        # Git file handling
 LICENSE               # MIT License
 README.md             # This file
```

---

## Configuration

### Training Configuration

Edit `configs/training_config.yaml`:

```yaml
model:
  type: "standard"  # standard, light, attention
  input_size: 256
  output_channels: 3

training:
  batch_size: 16
  num_epochs: 100
  learning_rate: 0.0001
  weight_decay: 0.0001
  
  optimizer:
    type: "adam"
    betas: [0.9, 0.999]
  
  scheduler:
    type: "cosine"
    t_max: 100
    eta_min: 0.00001

loss:
  type: "combined"
  ssim_weight: 0.7
  l1_weight: 0.3

data:
  augmentation: true
  num_workers: 4
  pin_memory: true

gpu:
  max_temperature: 80.0
  use_amp: true
  gradient_accumulation: 1
```

### Device Configuration

Edit `configs/device_config.yaml`:

```yaml
gpu:
  temperature:
    max_safe: 80.0
    warning: 75.0
    critical: 85.0
  
  memory:
    reserved_mb: 1024
    cache_clear_threshold: 0.9
  
  monitoring:
    interval_seconds: 10
    log_stats: true

fallback:
  allow_cpu: true
  require_permission: true
  save_checkpoint_on_fallback: true
```

---

## Edge Deployment

### Export to ONNX

```bash
python main.py export \
    --model-path checkpoints/unet_standard_best.pth \
    --output models/unet_standard.onnx \
    --input-shape 1 3 256 256 \
    --opset 11
```

### Convert to TensorRT (on Jetson)

```bash
# FP16 precision (recommended for Jetson Xavier/Orin)
python edge/convert_trt.py \
    --onnx-path models/unet_standard.onnx \
    --output models/unet_standard_fp16.trt \
    --precision fp16 \
    --workspace 4096

# INT8 precision (for Jetson Nano)
python edge/convert_trt.py \
    --onnx-path models/unet_light.onnx \
    --output models/unet_light_int8.trt \
    --precision int8 \
    --calibration-data data/calibration
```

### Run on Jetson

```python
from edge import JetsonInference

# Load TensorRT engine
inference = JetsonInference('models/unet_standard_fp16.trt')

# Enhance image
enhanced = inference.infer('underwater.jpg')
inference.save_result(enhanced, 'enhanced.jpg')

# Benchmark
stats = inference.benchmark(num_runs=100)
print(f"Average FPS: {stats['fps']:.2f}")
```

---

## Performance

### Inference Speed (FPS)

| Model | RTX 4050 | Jetson Xavier NX | Jetson Nano |
|-------|----------|------------------|-------------|
| UNet-Standard | 45 | 12 | 3 |
| UNet-Light | 120 | 35 | 10 |
| UNet-Attention | 30 | 8 | 2 |

### Quality Metrics (Average on EUVP Test Set)

| Model | PSNR (dB) | SSIM | Parameters |
|-------|-----------|------|------------|
| UNet-Standard | 28.5 | 0.912 | 31M |
| UNet-Light | 26.8 | 0.895 | 8M |
| UNet-Attention | 29.2 | 0.923 | 42M |

### Memory Usage

| Model | Training (Batch=16) | Inference (Single) |
|-------|---------------------|-------------------|
| UNet-Standard | 5.2 GB | 0.8 GB |
| UNet-Light | 2.8 GB | 0.3 GB |
| UNet-Attention | 6.5 GB | 1.1 GB |

---

## Testing

### Run All Tests

```bash
pytest tests/ -v
```

### Run Specific Tests

```bash
# Model architecture tests
pytest tests/test_models.py -v

# Inference pipeline tests
pytest tests/test_inference.py -v

# Training tests
pytest tests/test_training.py -v

# Utility tests
pytest tests/test_utils.py -v
```

### Coverage Report

```bash
pytest tests/ --cov=. --cov-report=html
open htmlcov/index.html
```

---

## Documentation

- **[Architecture Guide](docs/ARCHITECTURE.md)**: Detailed system architecture and design decisions
- **[Training Guide](docs/TRAINING_GUIDE.md)**: Best practices for training models
- **[Deployment Guide](docs/DEPLOYMENT_GUIDE.md)**: Step-by-step edge deployment instructions
- **[API Reference](docs/API_REFERENCE.md)**: Complete API documentation

---

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

**Please ensure:**

- All tests pass (`pytest tests/`)
- Code follows PEP 8 style guide
- Documentation is updated
- Commit messages are descriptive

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **U-Net Architecture**: Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation", MICCAI 2015
- **SSIM Loss**: Wang et al., "Image Quality Assessment: From Error Visibility to Structural Similarity", IEEE TIP 2004
- **Underwater Datasets**:
  - EUVP: Islam et al., "Fast Underwater Image Enhancement for Improved Visual Perception", IEEE RAL 2020
  - UIEB: Li et al., "An Underwater Image Enhancement Benchmark Dataset and Beyond", IEEE TIP 2020
- **PyTorch**: Facebook AI Research
- **NVIDIA**: CUDA, cuDNN, TensorRT frameworks

---

## Citation

If you use BlueDepth-Crescent in your research, please cite:

```bibtex
@software{bluedepth_crescent_2025,
  title={BlueDepth-Crescent: Underwater Vision Intelligence System},
  author={AbyssDrn},
  year={2025},
  url={https://github.com/AbyssDrn/BlueDepth-Crescent}
}
```

---