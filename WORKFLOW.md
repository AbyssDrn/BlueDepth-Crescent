#  BlueDepth-Crescent Workflow

## Complete Workflow

### 1. Setup (One-time)
```bash
# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Validate installation
python scripts/validate_installation.py

# Setup dataset structure
python scripts/setup_dataset.py
```

### 2. Prepare Data
```bash
# Place images in folders
data/
 hazy/    # Your underwater images
 clear/   # Ground truth (if available)
```

### 3. Train Model
```bash
# Quick training
python main.py train

# Custom training
python main.py train --model light --epochs 50 --batch_size 16
```

### 4. Monitor Training
```bash
# Option 1: Dashboard
python main.py dashboard

# Option 2: TensorBoard
tensorboard --logdir logs/
```

### 5. Inference
```bash
# Single image
python main.py enhance --input test.jpg --output enhanced.jpg

# Batch
python main.py batch --input_dir raw/ --output_dir enhanced/

# Video
python main.py video --input video.mp4 --output frame.jpg
```

### 6. Deploy to Edge
```bash
# Export to ONNX
python main.py export --model_path checkpoints/best.pth --output model.onnx

# Convert to TensorRT (on Jetson)
python edge/convert_trt.py

# Run on Jetson
python edge/jetson_inference.py
```

## Daily Usage

```bash
# Activate environment
source venv/bin/activate

# Enhance images
python main.py enhance --input my_image.jpg --output result.jpg

# Or use dashboard
python main.py dashboard
```

## Tips

- Always activate venv before running
- Check GPU temp: `nvidia-smi`
- Monitor training in dashboard
- Save checkpoints frequently
- Test on small dataset first
