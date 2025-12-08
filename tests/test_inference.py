"""
Tests for inference components
Tests: enhancer, classifier, batch_processor, video_processor
"""

import pytest
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.enhancer import UnderwaterEnhancer
from inference.classifier import UnderwaterImageClassifier
from inference.batch_processor import BatchProcessor
from models import UNetStandard, UnderwaterClassifier


class TestEnhancer:
    """Test underwater enhancer"""
    
    def test_enhancer_initialization(self):
        """Test enhancer initialization"""
        model = UNetStandard()
        enhancer = UnderwaterEnhancer(model=model)
        
        assert enhancer is not None
    
    def test_enhance_single_image(self, sample_image_path):
        """Test enhancing single image"""
        model = UNetStandard()
        enhancer = UnderwaterEnhancer(model=model)
        
        enhanced = enhancer.enhance(str(sample_image_path))
        
        assert enhanced is not None
        assert enhanced.shape[0] == 3  # RGB


class TestClassifier:
    """Test underwater classifier"""
    
    def test_classifier_initialization(self):
        """Test classifier initialization"""
        model = UnderwaterClassifier(num_classes=10)
        classifier = UnderwaterImageClassifier(model=model)
        
        assert classifier is not None
    
    def test_classify_single_image(self, sample_image_path):
        """Test classifying single image"""
        model = UnderwaterClassifier(num_classes=10)
        classifier = UnderwaterImageClassifier(model=model)
        
        class_id, confidence = classifier.classify(str(sample_image_path))
        
        assert 0 <= class_id < 10
        assert 0 <= confidence <= 1


class TestBatchProcessor:
    """Test batch processor"""
    
    def test_batch_processor_initialization(self):
        """Test batch processor initialization"""
        model = UNetStandard()
        processor = BatchProcessor(model=model)
        
        assert processor is not None


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
