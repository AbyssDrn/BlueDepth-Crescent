# BlueDepth-Crescent Dataset Structure

## Folder Organization

- **raw/**: Original underwater images (any format)
- **hazy/**: Training images (degraded underwater images)
- **clear/**: Training labels (enhanced/ground truth images)
- **processed/**: Preprocessed images ready for training
- **enhanced/**: Model output (enhanced images)
- **videos/**: Input videos for frame extraction
- **frames/**: Extracted video frames
- **test/**: Test images for evaluation

## Dataset Preparation

### Option 1: Use Your Own Data
1. Place your underwater images in `hazy/`
2. Place corresponding enhanced images in `clear/` (if available)
3. Or place any underwater images in `raw/` for unsupervised enhancement

### Option 2: Download Public Datasets
- **EUVP**: https://irvlab.cs.umn.edu/resources/euvp-dataset
- **UIEB**: https://li-chongyi.github.io/proj_benchmark.html
- **UFO-120**: http://irvlab.cs.umn.edu/resources/ufo-120-dataset

### Data Requirements
- **Minimum**: 5,000 images for basic training
- **Recommended**: 20,000+ images for best results
- **Formats**: JPG, PNG, JPEG
- **Resolution**: 256x256 to 1024x1024

### Video Processing
- Place videos (<1 minute) in `videos/`
- Run frame extraction: `python main.py video --input videos/sample.mp4`

## Notes
- Pair your hazy/clear images with matching filenames
- Use consistent naming: img_001.jpg in both folders
- Ensure image pairs have identical dimensions
