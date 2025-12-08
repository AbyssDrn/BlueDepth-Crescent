# Training Data Instructions

## Folder Purpose

### data/hazy/
- **What**: Degraded underwater images (training input)
- **Your data**: 1600 JPG images
- **Naming**: hazy_001.jpg, hazy_002.jpg, etc.

### data/clear/
- **What**: Enhanced/ground truth images (training target)
- **Your data**: 1600 JPG images
- **Naming**: Must match hazy images (clear_001.jpg, clear_002.jpg, etc.)
- **Important**: Each image in hazy/ must have a corresponding image in clear/

##  Your Current Status
- hazy/: 1600 images 
- clear/: 1600 images 
- Ready for training!

## Next Steps
1. Verify image pairs match (same filenames)
2. Start training: `python main.py train`
