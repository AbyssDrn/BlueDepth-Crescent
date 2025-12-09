#!/usr/bin/env python3
"""
Generate comprehensive .gitignore for BlueDepth-Crescent
Run this before committing to Git
"""

gitignore_content = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual Environment
venv/
env/
ENV/
.venv
virtualenv/

# PyTorch Models (LARGE FILES - DO NOT UPLOAD)
checkpoints/*.pth
*.pth
*.pt
*.ckpt
!checkpoints/README.md

# Data Files (LARGE FILES - DO NOT UPLOAD)
data/train/
data/test/
data/val/
data/raw/
data/hazy/
data/clear/
data/frames/
data/videos/
data/enhanced/
*.jpg
*.jpeg
*.png
*.bmp
*.gif
*.mp4
*.avi
*.mov
*.mkv
!docs/**/*.png
!ui/assets/**/*.png

# Results and Logs
results/
logs/
runs/
*.log
*.out

# Jupyter Notebooks
.ipynb_checkpoints/
*.ipynb

# IDE
.vscode/
.idea/
*.swp
*.swo
*~
.project
.pydevproject
.settings/

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db
desktop.ini

# Temporary files
*.tmp
*.bak
*.cache
*.temp
.pytest_cache/
.mypy_cache/
.coverage
htmlcov/

# Environment variables
.env
.env.local

# ONNX/TensorRT
*.onnx
*.trt
*.engine

# Compiled files
*.pyc
*.pyo
"""

with open('.gitignore', 'w') as f:
    f.write(gitignore_content.strip())

print(" Created .gitignore")
print("\nThis will prevent uploading:")
print("  - Virtual environment (venv/)")
print("  - Model checkpoints (*.pth)")
print("  - Training data (images, videos)")
print("  - Results and logs")
print("\n Safe to commit to Git now!")
