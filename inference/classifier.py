"""
Underwater Object Classification and Threat Assessment
Maritime security and reconnaissance
"""

import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from typing import Union, Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class ObjectClassifier:
    """
    Underwater Object Classifier with Threat Assessment
    
    Detects and classifies objects in underwater images with confidence scores
    and threat level evaluation for maritime security.
    """
    
    def __init__(
        self, 
        model_path: str, 
        device: str = 'cuda'
    ):
        """
        Initialize the object classifier
        
        Args:
            model_path: Path to trained classifier checkpoint
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load model
        from models import UnderwaterClassifier, OBJECT_CATEGORIES, THREAT_LEVELS
        
        self.categories = OBJECT_CATEGORIES
        self.threat_levels = THREAT_LEVELS
        
        self.model = UnderwaterClassifier(
            num_classes=len(OBJECT_CATEGORIES),
            input_size=224
        )
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        logger.info(f"ObjectClassifier initialized on {self.device}")
    
    @torch.no_grad()
    def classify(
        self, 
        image: Union[str, Image.Image, np.ndarray],
        top_k: int = 5
    ) -> Dict[str, any]:
        """
        Classify object in underwater image with threat assessment
        
        Args:
            image: Input image (path, PIL Image, or numpy array)
            top_k: Number of top predictions to return
            
        Returns:
            Dictionary containing:
            - class_name: Predicted class
            - confidence: Confidence score
            - threat_level: Threat assessment
            - threat_confidence: Threat confidence
            - all_predictions: Top-k predictions with scores
        """
        # Load and convert image
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Transform and add batch dimension
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Get predictions
        result = self.model.predict_single(input_tensor[0])
        
        # Get top-k class predictions
        class_logits, threat_logits = self.model(input_tensor)
        class_probs = torch.softmax(class_logits, dim=1).squeeze(0).cpu().numpy()
        
        # Get top predictions
        top_indices = np.argsort(class_probs)[::-1][:top_k]
        top_predictions = [
            {
                'class_id': int(idx),
                'class_name': self.categories[idx]['name'],
                'confidence': float(class_probs[idx]),
                'description': self.categories[idx]['description']
            }
            for idx in top_indices
        ]
        
        return {
            'class_name': result['class_name'],
            'class_id': result['class_id'],
            'confidence': result['class_confidence'],
            'threat_level': result['threat_name'],
            'threat_confidence': result['threat_confidence'],
            'description': result['threat_description'],
            'top_predictions': top_predictions
        }
    
    def classify_batch(
        self, 
        images: List[Union[str, Image.Image, np.ndarray]]
    ) -> List[Dict[str, any]]:
        """
        Classify multiple images
        
        Args:
            images: List of images to classify
            
        Returns:
            List of classification results
        """
        results = []
        for img in images:
            try:
                result = self.classify(img)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to classify image: {e}")
                results.append(None)
        
        return results
    
    def get_threat_statistics(
        self, 
        results: List[Dict[str, any]]
    ) -> Dict[str, int]:
        """
        Analyze threat levels from batch classification results
        
        Args:
            results: List of classification results
            
        Returns:
            Dictionary with threat level counts
        """
        threat_counts = {level: 0 for level in self.threat_levels.keys()}
        
        for result in results:
            if result is not None:
                threat_level = result.get('threat_level', 'NONE')
                threat_counts[threat_level] = threat_counts.get(threat_level, 0) + 1
        
        return threat_counts
    
    def get_categories(self) -> Dict[int, Dict[str, str]]:
        """Get all object categories"""
        return self.categories
    
    def get_threat_levels(self) -> Dict[str, Dict[str, str]]:
        """Get all threat levels"""
        return self.threat_levels


class EnhancedClassifier:
    """
    Combined Enhancement + Classification Pipeline
    
    First enhances underwater images, then classifies objects with threat assessment.
    """
    
    def __init__(
        self,
        enhancer_path: str,
        classifier_path: str,
        model_type: str = 'unet_standard',
        device: str = 'cuda'
    ):
        """
        Initialize combined pipeline
        
        Args:
            enhancer_path: Path to enhancement model checkpoint
            classifier_path: Path to classifier model checkpoint
            model_type: Enhancement model type
            device: Device to run inference on
        """
        from .enhancer import ImageEnhancer
        
        self.enhancer = ImageEnhancer(enhancer_path, model_type, device)
        self.classifier = ObjectClassifier(classifier_path, device)
        
        logger.info("EnhancedClassifier pipeline initialized")
    
    def process(
        self, 
        image: Union[str, Image.Image, np.ndarray],
        return_enhanced: bool = False
    ) -> Dict[str, any]:
        """
        Enhance image and classify objects
        
        Args:
            image: Input image
            return_enhanced: If True, include enhanced image in result
            
        Returns:
            Dictionary with enhancement and classification results
        """
        # Enhance image
        enhanced_image = self.enhancer.enhance_image(image)
        
        # Classify enhanced image
        classification = self.classifier.classify(enhanced_image)
        
        result = {
            'classification': classification
        }
        
        if return_enhanced:
            result['enhanced_image'] = enhanced_image
        
        return result
