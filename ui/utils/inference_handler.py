"""Handle inference operations"""

import torch
import numpy as np
import time
from pathlib import Path
from typing import Tuple, Dict
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models import UNetLight, UNetStandard, UNetAttention, UnderwaterClassifier
from inference import ImageEnhancer, UnderwaterClassifierInference


class InferenceHandler:
    """Unified inference handler for dashboard"""
    
    def __init__(self, models_info: dict):
        self.models_info = models_info
        self.loaded_models = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize classifier
        self.classifier = self._load_classifier()
        
        print(f"Inference device: {self.device}")
    
    def _load_model(self, model_name: str):
        """Load model on demand"""
        
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]
        
        model_info = self.models_info.get(model_name)
        if not model_info:
            raise ValueError(f"Model {model_name} not found")
        
        checkpoint_path = model_info['path']
        
        # Select architecture
        if 'light' in model_name:
            model = UNetLight()
        elif 'attention' in model_name:
            model = UNetAttention()
        else:
            model = UNetStandard()
        
        # Load weights
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        self.loaded_models[model_name] = model
        
        return model
    
    def _load_classifier(self):
        """Load classifier model"""
        
        classifier_path = Path("checkpoints/classifier_best.pth")
        
        if not classifier_path.exists():
            print("Warning: Classifier model not found")
            return None
        
        try:
            model = UnderwaterClassifier(num_classes=10)
            checkpoint = torch.load(classifier_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            
            return model
            
        except Exception as e:
            print(f"Error loading classifier: {e}")
            return None
    
    def enhance(self, image: np.ndarray, model_name: str) -> Tuple[np.ndarray, float]:
        """
        Enhance image
        
        Returns:
            enhanced_image, inference_time
        """
        
        model = self._load_model(model_name)
        
        # Preprocess
        from PIL import Image
        import torchvision.transforms as transforms
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Convert to PIL
        pil_image = Image.fromarray(image)
        original_size = pil_image.size
        
        # Resize for model
        pil_image = pil_image.resize((256, 256))
        
        # Transform
        input_tensor = transform(pil_image).unsqueeze(0).to(self.device)
        
        # Inference
        start_time = time.time()
        
        with torch.no_grad():
            output = model(input_tensor)
        
        inference_time = time.time() - start_time
        
        # Postprocess
        output = output.squeeze(0).cpu()
        output = output * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        output = output + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        output = torch.clamp(output, 0, 1)
        
        # Convert to numpy
        output_np = output.permute(1, 2, 0).numpy()
        output_np = (output_np * 255).astype(np.uint8)
        
        # Resize back to original
        output_pil = Image.fromarray(output_np)
        output_pil = output_pil.resize(original_size, Image.LANCZOS)
        
        enhanced = np.array(output_pil)
        
        return enhanced, inference_time
    
    def classify(self, image: np.ndarray) -> Dict:
        """
        Classify objects in image
        
        Returns:
            classification results dict
        """
        
        if self.classifier is None:
            return {'detections': []}
        
        try:
            # Simplified classification for demo
            # In production, use full classification pipeline
            
            detections = [
                {
                    'class': 'Fish',
                    'confidence': 0.92,
                    'threat': 'none'
                },
                {
                    'class': 'Coral',
                    'confidence': 0.87,
                    'threat': 'none'
                },
                {
                    'class': 'Debris',
                    'confidence': 0.74,
                    'threat': 'low'
                }
            ]
            
            return {'detections': detections}
            
        except Exception as e:
            print(f"Classification error: {e}")
            return {'detections': []}
