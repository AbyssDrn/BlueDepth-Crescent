"""
PyTorch Datasets for Underwater Image Enhancement
Maritime Security and Reconnaissance System
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
import torchvision.transforms as transforms
import random


class UnderwaterDataset(Dataset):
    """
    Paired underwater image dataset
    Hazy/degraded images paired with clear/enhanced ground truth
    """
    
    def __init__(
        self,
        hazy_dir: str,
        clear_dir: str,
        img_size: int = 256,
        transform: Optional[transforms.Compose] = None,
        augment: bool = True
    ):
        self.hazy_dir = Path(hazy_dir)
        self.clear_dir = Path(clear_dir)
        self.img_size = img_size
        self.augment = augment
        
        # Find all images
        self.hazy_images = sorted(
            list(self.hazy_dir.glob('*.jpg')) +
            list(self.hazy_dir.glob('*.png')) +
            list(self.hazy_dir.glob('*.jpeg'))
        )
        
        self.clear_images = sorted(
            list(self.clear_dir.glob('*.jpg')) +
            list(self.clear_dir.glob('*.png')) +
            list(self.clear_dir.glob('*.jpeg'))
        )
        
        # Validation
        assert len(self.hazy_images) > 0, f"No images found in {hazy_dir}"
        assert len(self.clear_images) > 0, f"No images found in {clear_dir}"
        assert len(self.hazy_images) == len(self.clear_images), \
            f"Mismatch: {len(self.hazy_images)} hazy vs {len(self.clear_images)} clear images"
        
        # Transforms
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor()
            ])
        else:
            self.transform = transform
        
        # Augmentation transforms
        if self.augment:
            self.aug_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
            ])
    
    def __len__(self):
        return len(self.hazy_images)
    
    def __getitem__(self, idx):
        """Load and transform image pair"""
        hazy_img = Image.open(self.hazy_images[idx]).convert('RGB')
        clear_img = Image.open(self.clear_images[idx]).convert('RGB')
        
        # Apply augmentation to both images consistently
        if self.augment:
            seed = random.randint(0, 2**32)
            random.seed(seed)
            torch.manual_seed(seed)
            hazy_img = self.aug_transform(hazy_img)
            
            random.seed(seed)
            torch.manual_seed(seed)
            clear_img = self.aug_transform(clear_img)
        
        # Convert to tensors
        hazy_tensor = self.transform(hazy_img)
        clear_tensor = self.transform(clear_img)
        
        return hazy_tensor, clear_tensor


class VideoFrameDataset(Dataset):
    """
    Dataset for video frame extraction and processing
    Used for temporal underwater surveillance
    """
    
    def __init__(self, frames_dir: str, img_size: int = 256):
        self.frames_dir = Path(frames_dir)
        self.img_size = img_size
        
        self.frames = sorted(
            list(self.frames_dir.glob('*.jpg')) +
            list(self.frames_dir.glob('*.png'))
        )
        
        assert len(self.frames) > 0, f"No frames found in {frames_dir}"
        
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        """Load and transform single frame"""
        frame = Image.open(self.frames[idx]).convert('RGB')
        return self.transform(frame), str(self.frames[idx])


class ClassificationDataset(Dataset):
    """
    Dataset for underwater object classification
    Includes class labels and threat levels
    """
    
    def __init__(
        self,
        images_dir: str,
        labels_file: str,
        img_size: int = 224,
        transform: Optional[transforms.Compose] = None
    ):
        self.images_dir = Path(images_dir)
        self.img_size = img_size
        
        # Load labels
        self.samples = self._load_labels(labels_file)
        
        # Transforms
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
    
    def _load_labels(self, labels_file: str) -> List[Tuple[str, int, int]]:
        """Load image paths with class and threat labels"""
        samples = []
        with open(labels_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 3:
                    image_name = parts[0]
                    class_id = int(parts[1])
                    threat_id = int(parts[2])
                    samples.append((image_name, class_id, threat_id))
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Load image with labels"""
        image_name, class_id, threat_id = self.samples[idx]
        image_path = self.images_dir / image_name
        
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image)
        
        return image_tensor, class_id, threat_id
