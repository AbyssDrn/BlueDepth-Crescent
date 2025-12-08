#!/bin/bash
# BlueDepth-Crescent: Train all models sequentially
# Usage: bash scripts/train_all_models.sh

set -e  # Exit on error

echo "========================================="
echo "  BlueDepth-Crescent - Train All Models"
echo "========================================="
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if data exists
if [ ! -d "data/train" ]; then
    echo "Error: Training data not found!"
    echo "Run: python scripts/setup_dataset.py"
    exit 1
fi

# Device config (change this for Jetson Nano)
DEVICE_CONFIG="configs/rtx4050.yaml"

echo -e "${BLUE}Training device: RTX 4050${NC}"
echo ""

# Model 1: UNet Light
echo -e "${GREEN}[1/3] Training UNet Light...${NC}"
python -m training.train_unet \
    --config configs/model_light.yaml \
    --device-config $DEVICE_CONFIG \
    --epochs 100

# Model 2: UNet Standard
echo -e "${GREEN}[2/3] Training UNet Standard...${NC}"
python -m training.train_unet \
    --config configs/model_standard.yaml \
    --device-config $DEVICE_CONFIG \
    --epochs 100

# Model 3: UNet Attention
echo -e "${GREEN}[3/3] Training UNet Attention...${NC}"
python -m training.train_unet \
    --config configs/model_attention.yaml \
    --device-config $DEVICE_CONFIG \
    --epochs 100

echo ""
echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}  All models trained successfully!${NC}"
echo -e "${GREEN}=========================================${NC}"
echo ""
echo "Run benchmark: python scripts/benchmark.py"
