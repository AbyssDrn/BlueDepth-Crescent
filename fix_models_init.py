#!/usr/bin/env python3
"""
Fix models/__init__.py
"""

content = '''"""BlueDepth-Crescent Models"""
from .base_model import BaseModel
from .unet_standard import UNetStandard
from .unet_light import UNetLight
from .unet_attention import UNetAttention
from .classifier import UnderwaterClassifier

__all__ = ['BaseModel', 'UNetStandard', 'UNetLight', 'UNetAttention', 'UnderwaterClassifier']
'''

with open("models/__init__.py", "w", encoding="utf-8") as f:
    f.write(content)

print(" Fixed models/__init__.py")
print()
print("Now check if the models exist:")

from pathlib import Path

model_files = [
    "models/base_model.py",
    "models/unet_standard.py", 
    "models/unet_light.py",
    "models/unet_attention.py",
    "models/classifier.py"
]

all_exist = True
for file in model_files:
    exists = Path(file).exists()
    status = "" if exists else ""
    print(f"{status} {file}")
    if not exists:
        all_exist = False

print()
if all_exist:
    print(" All model files exist!")
    print()
    print("Test import:")
    try:
        from models import UNetStandard, UNetLight, UNetAttention
        print(" Import successful!")
    except Exception as e:
        print(f" Import failed: {e}")
else:
    print(" Some model files are missing!")
    print()
    print("You need to run the project generator scripts:")
    print("   python generate_project.py")
    print("   python generate_part2.py")
    print("   python generate_part3.py")