"""
Comprehensive tests for all model architectures
Tests: UNetLight, UNetStandard, UNetAttention, UnderwaterClassifier
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import UNetLight, UNetStandard, UNetAttention, UnderwaterClassifier
from models.base_model import BaseUNet


class TestUNetLight:
    """Test UNet Light architecture"""
    
    def test_initialization(self):
        """Test model initialization"""
        model = UNetLight()
        assert isinstance(model, nn.Module)
        assert isinstance(model, BaseUNet)
    
    def test_forward_pass(self, dummy_tensor):
        """Test forward pass with single image"""
        model = UNetLight()
        model.eval()
        
        with torch.no_grad():
            output = model(dummy_tensor)
        
        assert output.shape == dummy_tensor.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_forward_pass_batch(self, dummy_batch):
        """Test forward pass with batch"""
        model = UNetLight()
        model.eval()
        
        with torch.no_grad():
            output = model(dummy_batch)
        
        assert output.shape == dummy_batch.shape
    
    def test_parameter_count(self):
        """Test parameter count is reasonable for light model"""
        model = UNetLight()
        total_params = sum(p.numel() for p in model.parameters())
        
        # UNet Light should have ~350K parameters
        assert 300_000 < total_params < 500_000, f"Expected ~350K params, got {total_params}"
    
    def test_gradient_flow(self, dummy_tensor):
        """Test gradients flow correctly"""
        model = UNetLight()
        model.train()
        
        output = model(dummy_tensor)
        loss = output.sum()
        loss.backward()
        
        # Check at least some gradients are non-zero
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 
                      for p in model.parameters())
        assert has_grad, "No gradients computed"
    
    def test_different_input_sizes(self):
        """Test model handles different input sizes"""
        model = UNetLight()
        model.eval()
        
        sizes = [128, 256, 512]
        for size in sizes:
            x = torch.randn(1, 3, size, size)
            with torch.no_grad():
                output = model(x)
            assert output.shape == x.shape


class TestUNetStandard:
    """Test UNet Standard architecture"""
    
    def test_initialization(self):
        """Test model initialization"""
        model = UNetStandard()
        assert isinstance(model, BaseUNet)
    
    def test_forward_pass(self, dummy_tensor):
        """Test forward pass"""
        model = UNetStandard()
        model.eval()
        
        with torch.no_grad():
            output = model(dummy_tensor)
        
        assert output.shape == dummy_tensor.shape
    
    def test_parameter_count(self):
        """Test parameter count"""
        model = UNetStandard()
        total_params = sum(p.numel() for p in model.parameters())
        
        # UNet Standard should have ~7.8M parameters
        assert 7_000_000 < total_params < 9_000_000, f"Expected ~7.8M params, got {total_params}"
    
    def test_output_range(self, dummy_tensor):
        """Test output is in valid range"""
        model = UNetStandard()
        model.eval()
        
        with torch.no_grad():
            output = model(dummy_tensor)
        
        # Output should be roughly in [0, 1] range due to sigmoid/tanh
        assert output.min() >= -0.1, "Output values too negative"
        assert output.max() <= 1.1, "Output values too positive"
    
    def test_save_load(self, tmp_path):
        """Test model save and load"""
        model1 = UNetStandard()
        checkpoint_path = tmp_path / "test_checkpoint.pth"
        
        # Save
        torch.save(model1.state_dict(), checkpoint_path)
        
        # Load
        model2 = UNetStandard()
        model2.load_state_dict(torch.load(checkpoint_path))
        
        # Verify same weights
        x = torch.randn(1, 3, 256, 256)
        model1.eval()
        model2.eval()
        
        with torch.no_grad():
            out1 = model1(x)
            out2 = model2(x)
        
        assert torch.allclose(out1, out2, atol=1e-6)


class TestUNetAttention:
    """Test UNet Attention architecture"""
    
    def test_initialization(self):
        """Test model initialization with attention"""
        model = UNetAttention()
        assert isinstance(model, BaseUNet)
    
    def test_forward_pass(self, dummy_tensor):
        """Test forward pass with attention"""
        model = UNetAttention()
        model.eval()
        
        with torch.no_grad():
            output = model(dummy_tensor)
        
        assert output.shape == dummy_tensor.shape
    
    def test_parameter_count(self):
        """Test parameter count (should be > UNetStandard)"""
        model = UNetAttention()
        total_params = sum(p.numel() for p in model.parameters())
        
        # UNet Attention should have ~8.2M parameters
        assert 7_500_000 < total_params < 9_500_000, f"Expected ~8.2M params, got {total_params}"
    
    def test_attention_mechanism(self, dummy_batch):
        """Test attention improves focus"""
        model = UNetAttention()
        model.eval()
        
        with torch.no_grad():
            output = model(dummy_batch)
        
        # Check output has reasonable statistics
        assert output.std() > 0.01, "Output has no variation (attention might be broken)"
    
    def test_memory_efficiency(self):
        """Test model doesn't use excessive memory"""
        model = UNetAttention()
        
        # Should be able to process 256x256 image without issues
        x = torch.randn(1, 3, 256, 256)
        
        try:
            with torch.no_grad():
                _ = model(x)
            success = True
        except RuntimeError:
            success = False
        
        assert success, "Model uses too much memory"


class TestUnderwaterClassifier:
    """Test Underwater Classifier"""
    
    def test_initialization_default(self):
        """Test default initialization"""
        model = UnderwaterClassifier()
        assert isinstance(model, nn.Module)
    
    def test_initialization_custom_classes(self):
        """Test initialization with custom number of classes"""
        num_classes = 5
        model = UnderwaterClassifier(num_classes=num_classes)
        
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (1, num_classes)
    
    def test_forward_pass(self):
        """Test forward pass"""
        model = UnderwaterClassifier(num_classes=10)
        model.eval()
        
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (2, 10)
    
    def test_output_logits(self):
        """Test output is valid logits"""
        model = UnderwaterClassifier(num_classes=5)
        model.eval()
        
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(x)
        
        # Logits can be any real number
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_softmax_output(self):
        """Test softmax gives valid probabilities"""
        model = UnderwaterClassifier(num_classes=5)
        model.eval()
        
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
        
        # Probabilities should sum to 1
        assert torch.allclose(probs.sum(dim=1), torch.tensor([1.0]), atol=1e-5)
        assert (probs >= 0).all() and (probs <= 1).all()


class TestModelComparison:
    """Compare all models"""
    
    def test_parameter_ordering(self):
        """Test models have correct parameter ordering"""
        light = UNetLight()
        standard = UNetStandard()
        attention = UNetAttention()
        
        light_params = sum(p.numel() for p in light.parameters())
        standard_params = sum(p.numel() for p in standard.parameters())
        attention_params = sum(p.numel() for p in attention.parameters())
        
        assert light_params < standard_params < attention_params, \
            "Models should have increasing parameter counts"
    
    def test_all_models_same_output_shape(self, dummy_tensor):
        """Test all UNet models produce same output shape"""
        models = [UNetLight(), UNetStandard(), UNetAttention()]
        
        for model in models:
            model.eval()
            with torch.no_grad():
                output = model(dummy_tensor)
            assert output.shape == dummy_tensor.shape
    
    @pytest.mark.slow
    def test_inference_speed_comparison(self, dummy_batch):
        """Test relative inference speeds"""
        import time
        
        models = {
            'light': UNetLight(),
            'standard': UNetStandard(),
            'attention': UNetAttention()
        }
        
        times = {}
        
        for name, model in models.items():
            model.eval()
            
            # Warmup
            with torch.no_grad():
                _ = model(dummy_batch)
            
            # Measure
            start = time.time()
            with torch.no_grad():
                for _ in range(10):
                    _ = model(dummy_batch)
            times[name] = time.time() - start
        
        # Light should be fastest
        assert times['light'] < times['standard']
        assert times['light'] < times['attention']


class TestModelRobustness:
    """Test model robustness"""
    
    def test_zero_input(self):
        """Test models handle zero input"""
        model = UNetStandard()
        model.eval()
        
        x = torch.zeros(1, 3, 256, 256)
        with torch.no_grad():
            output = model(x)
        
        assert not torch.isnan(output).any()
    
    def test_ones_input(self):
        """Test models handle ones input"""
        model = UNetStandard()
        model.eval()
        
        x = torch.ones(1, 3, 256, 256)
        with torch.no_grad():
            output = model(x)
        
        assert not torch.isnan(output).any()
    
    def test_extreme_values(self):
        """Test models handle extreme input values"""
        model = UNetStandard()
        model.eval()
        
        # Test with very large values
        x = torch.randn(1, 3, 256, 256) * 100
        with torch.no_grad():
            output = model(x)
        
        assert not torch.isnan(output).any()
    
    def test_batch_size_one(self, dummy_tensor):
        """Test batch size 1"""
        model = UNetStandard()
        model.eval()
        
        with torch.no_grad():
            output = model(dummy_tensor)
        
        assert output.shape[0] == 1
    
    def test_batch_size_varied(self):
        """Test various batch sizes"""
        model = UNetLight()
        model.eval()
        
        for batch_size in [1, 2, 4, 8]:
            x = torch.randn(batch_size, 3, 256, 256)
            with torch.no_grad():
                output = model(x)
            assert output.shape[0] == batch_size


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
