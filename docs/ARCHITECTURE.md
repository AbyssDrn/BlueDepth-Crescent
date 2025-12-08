# BlueDepth-Crescent Architecture

**Version:** 1.0.0  
**Last Updated:** December 8, 2025

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture Diagram](#architecture-diagram)
3. [Component Design](#component-design)
4. [Model Architectures](#model-architectures)
5. [Data Flow](#data-flow)
6. [Deployment Architecture](#deployment-architecture)
7. [Performance Considerations](#performance-considerations)

---

## System Overview

BlueDepth-Crescent is a comprehensive underwater image enhancement and object classification system designed for:

- **Maritime Security**: AUV/ROV vision systems
- **Marine Research**: Scientific imaging
- **Edge Deployment**: Jetson Nano, mobile devices
- **Cloud Processing**: Batch processing pipelines

### Design Principles

1. **Modularity**: Each component can be used independently
2. **Scalability**: From edge devices to cloud clusters
3. **Extensibility**: Easy to add new models and features
4. **Performance**: Optimized for real-time processing

---

## Architecture Diagram


 BlueDepth-Crescent System 



  
  
 UI   Inference   Training 
 (Gradio)  Layer   Pipeline 
  
  



 Models Layer 

 - UNetLight 
 - UNetStandard 
 - UNetAttention 
 - Classifier 



 Data Layer 

 - Dataset 
 - Transforms 
 - Data Loaders 



 Utils Layer 

 - Metrics 
 - Visualization 
 - Device Manager 
 - Logger 



---

## Component Design

### 1. Models (`models/`)

**Purpose:** Core neural network architectures

**Structure:**
models/
 init.py # Public API
 base_model.py # Abstract base class
 unet_light.py # Lightweight U-Net (~350K params)
 unet_standard.py # Standard U-Net (~7.8M params)
 unet_attention.py # Attention U-Net (~8.2M params)
 classifier.py # Object classifier

tex

**Design Pattern:** Template Method + Factory

Base class defines interface
class BaseModel(nn.Module):
def forward(self, x): raise NotImplementedError
def load_checkpoint(self, path): ...
def save_checkpoint(self, path): ...

Concrete implementations
class UNetStandard(BaseModel):
def forward(self, x):
# Specific architecture



---

### 2. Inference (`inference/`)

**Purpose:** Production inference pipelines

**Structure:**
inference/
 init.py
 enhancer.py # Single image enhancement
 batch_processor.py # Batch processing
 video_processor.py # Video enhancement
 classifier.py # Classification inference



**Key Features:**
- **Preprocessing pipeline:** Normalization, resizing
- **Postprocessing:** Denormalization, color correction
- **Batching:** Automatic batch size optimization
- **Caching:** Model and data caching

---

### 3. Training (`training/`)

**Purpose:** Model training pipelines

**Structure:**
training/
 init.py
 train_unet.py # U-Net trainer
 losses.py # Loss functions
 callbacks.py # Training callbacks (coming soon)



**Training Loop:**
Load data → DataLoader

Forward pass → Model

Calculate loss → Loss function

Backward pass → Optimizer

Update metrics → Metrics tracker

Save checkpoint → Best model only

Log results → TensorBoard/console



---

### 4. Data (`data/`)

**Purpose:** Data loading and preprocessing

**Structure:**
data/
 init.py
 dataset.py # PyTorch Dataset
 transforms.py # Image augmentations
 README.md # Data format guide



**Data Pipeline:**
Raw Image → Transform → Normalize → Batch → Model



**Augmentations:**
- Random horizontal flip
- Random rotation
- Color jitter
- Gaussian noise
- Underwater-specific augmentations

---

### 5. Utils (`utils/`)

**Purpose:** Shared utilities

**Structure:**
utils/
 init.py
 metrics.py # PSNR, SSIM, UIQM
 visualization.py # Plotting functions
 device_manager.py # GPU management
 logger.py # Logging utilities



---

### 6. Edge Deployment (`edge/`)

**Purpose:** Edge device optimization

**Structure:**
edge/
 init.py
 export.py # ONNX/TensorRT export
 quantization.py # Model quantization
 jetson_optimizer.py # Jetson-specific optimization



**Deployment Targets:**
- Jetson Nano (TensorRT)
- Jetson Xavier NX
- Mobile devices (ONNX)
- Web browsers (ONNX.js)

---

## Model Architectures

### UNetStandard Architecture

Input (3×256×256)
↓

 Encoder Block  64 channels
 ↓ MaxPool 

 Encoder Block  128 channels
 ↓ MaxPool 

 Encoder Block  256 channels
 ↓ MaxPool 

 Encoder Block  512 channels (Bottleneck)
 ↑ UpSample 

 Decoder Block  256 channels + skip
 ↑ UpSample 

 Decoder Block  128 channels + skip
 ↑ UpSample 

 Decoder Block  64 channels + skip
 ↓ 

↓
Output (3×256×256)


**Key Components:**
- **Encoder:** Progressive downsampling
- **Bottleneck:** Feature extraction
- **Decoder:** Progressive upsampling
- **Skip Connections:** Preserve spatial information

---

### UNetAttention Architecture

Same as UNetStandard + Attention Gates

Decoder Block with Attention:

 Skip Feature



 Attention  ← Gating signal from previous layer
 Gate 



 Concatenate 



 Conv Block 



**Benefits:**
- Focus on relevant regions
- Better performance on complex scenes
- Improved edge preservation

---

### Classifier Architecture

Input (3×224×224)
↓

 Backbone  ResNet50 / EfficientNet-B0
 (pretrained) 

↓

 Global Pool 

↓

 FC Layer  → num_classes

↓
Predictions


---

## Data Flow

### Training Data Flow

Dataset Directory
 hazy/
  img_001.jpg
  ...
 clean/
 img_001.jpg
 ...

UnderwaterDataset

Load image pairs

Apply transforms

Normalize

DataLoader

Batch images

Shuffle (training)

Multi-threading

Training Loop

Forward pass

Loss calculation

Backward pass

Optimization

Checkpoint

Save best model

Save metrics

Save history


### Inference Data Flow

Input Image
↓

Preprocessing

Resize to 256×256

Normalize [0-1]

Convert to tensor
↓

Model Forward

GPU inference

FP16 (optional)
↓

Postprocessing

Denormalize

Resize to original

Convert to numpy
↓

Output Image


---

## Deployment Architecture

### Cloud Deployment


 Load Balancer (nginx) 



  
  
 API-1   API-2   API-3  FastAPI Instances
  
  



 Model Storage  Shared checkpoint dir
 (NFS/S3/GCS) 



**Features:**
- Horizontal scaling
- Load balancing
- Shared model storage
- Health monitoring

---

### Edge Deployment


 AUV/ROV Device 

 - Jetson Nano 
 - Camera Input 
 - Local Processing 
 - TensorRT Optimization 


 WiFi/Ethernet (optional)


 Control Center 
 - Monitoring 
 - Data Collection 



**Optimization:**
- TensorRT INT8 quantization
- Model pruning (40% reduction)
- Batch size = 1
- Resolution 256×256

---

## Performance Considerations

### Memory Usage

| Model          | Parameters | GPU Memory (FP32) | GPU Memory (FP16) |
|----------------|------------|-------------------|-------------------|
| UNetLight      | 350K       | 400 MB           | 250 MB           |
| UNetStandard   | 7.8M       | 2.1 GB           | 1.2 GB           |
| UNetAttention  | 8.2M       | 2.3 GB           | 1.4 GB           |
| Classifier     | 25M        | 1.8 GB           | 1.0 GB           |

### Inference Speed (RTX 4050)

| Model          | FP32 (ms) | FP16 (ms) | FPS (FP16) |
|----------------|-----------|-----------|------------|
| UNetLight      | 8.5       | 4.2       | 238        |
| UNetStandard   | 18.3      | 9.1       | 110        |
| UNetAttention  | 22.7      | 11.3      | 88         |

**Resolution:** 256×256

### Optimization Techniques

1. **Mixed Precision Training**
   - Use FP16 for forward/backward
   - FP32 for gradient accumulation
   - 2× speedup, 50% memory reduction

2. **Gradient Accumulation**
   - Simulate larger batch sizes
   - Accumulate gradients over N steps
   - Useful for limited VRAM

3. **Model Checkpointing**
   - Trade compute for memory
   - Recompute activations during backward
   - Enable larger models

4. **Data Loading**
   - Pin memory for GPU transfer
   - Multi-worker data loading
   - Prefetch next batch

---

## Security Considerations

1. **Model Protection**
   - Encrypt checkpoints
   - Watermark models
   - Access control

2. **Input Validation**
   - Image format verification
   - Size limits
   - Malicious input detection

3. **API Security**
   - Authentication tokens
   - Rate limiting
   - Input sanitization

---

## Future Enhancements

1. **Multi-GPU Support**
   - DataParallel
   - DistributedDataParallel
   - Model parallelism

2. **Real-Time Video**
   - Temporal consistency
   - Frame buffering
   - Optical flow integration

3. **Active Learning**
   - Uncertainty estimation
   - Hard example mining
   - Continuous improvement

---

**Last Updated:** December 8, 2025