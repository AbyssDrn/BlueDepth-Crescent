"""Complete Model Testing Script
Tests all models and verifies functionality
"""

import torch
import sys
from pathlib import Path

# Add models directory to path
sys.path.insert(0, str(Path(__file__).parent))

from models import (
    UNetStandard,
    UNetLight,
    UNetAttention,
    UnderwaterClassifier,
    get_model,
    list_models,
    get_model_info,
    OBJECT_CATEGORIES,
    THREAT_LEVELS
)

def print_header(text):
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def test_base_functionality():
    """Test base model functionality"""
    print_header("Testing Base Model Functionality")
    
    model = UNetStandard()
    
    # Test model info
    model.print_model_info()
    
    # Test device management
    if torch.cuda.is_available():
        model.to_device(torch.device('cuda'))
        print(f"Model on device: {model.device}")
        model.to_device(torch.device('cpu'))
    
    print("Base functionality: PASSED")

def test_unet_standard():
    """Test UNet Standard"""
    print_header("Testing UNet Standard")
    
    model = UNetStandard(n_channels=3, n_classes=3)
    model.eval()
    
    # Test forward pass
    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        output = model(x)
    
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
    
    assert output.shape == x.shape, "Output shape mismatch!"
    assert output.min() >= 0 and output.max() <= 1, "Output not in [0,1] range!"
    
    print("UNet Standard: PASSED")

def test_unet_light():
    """Test UNet Light"""
    print_header("Testing UNet Light")
    
    model = UNetLight(n_channels=3, n_classes=3)
    model.eval()
    
    # Test forward pass
    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        output = model(x)
    
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameters:   {model.count_parameters():,}")
    print(f"Model size:   {model.get_model_size_mb():.2f} MB")
    
    assert output.shape == x.shape, "Output shape mismatch!"
    assert output.min() >= 0 and output.max() <= 1, "Output not in [0,1] range!"
    
    print("UNet Light: PASSED")

def test_unet_attention():
    """Test UNet Attention"""
    print_header("Testing UNet Attention")
    
    model = UNetAttention(n_channels=3, n_classes=3)
    model.eval()
    
    # Test forward pass
    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        output = model(x)
    
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameters:   {model.count_parameters():,}")
    
    assert output.shape == x.shape, "Output shape mismatch!"
    assert output.min() >= 0 and output.max() <= 1, "Output not in [0,1] range!"
    
    print("UNet Attention: PASSED")

def test_classifier():
    """Test Underwater Classifier"""
    print_header("Testing Underwater Classifier")
    
    model = UnderwaterClassifier(num_classes=15, input_size=224)
    model.eval()
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        class_logits, threat_logits = model(x)
    
    print(f"Input shape:         {x.shape}")
    print(f"Class logits shape:  {class_logits.shape}")
    print(f"Threat logits shape: {threat_logits.shape}")
    print(f"Parameters:          {model.count_parameters():,}")
    
    assert class_logits.shape == (2, 15), "Class output shape mismatch!"
    assert threat_logits.shape == (2, 5), "Threat output shape mismatch!"
    
    # Test prediction
    result = model.predict_single(x[0])
    
    print(f"\nPrediction Result:")
    print(f"  Class: {result['class_name']}")
    print(f"  Confidence: {result['class_confidence']:.2%}")
    print(f"  Threat Level: {result['threat_name']}")
    print(f"  Threat Confidence: {result['threat_confidence']:.2%}")
    print(f"  Description: {result['threat_description']}")
    
    print("Classifier: PASSED")

def test_object_categories():
    """Test object categories and threat levels"""
    print_header("Testing Object Categories")
    
    print(f"Total categories: {len(OBJECT_CATEGORIES)}")
    print(f"Threat levels: {len(THREAT_LEVELS)}")
    
    print("\nCategory Overview:")
    for cat_id, info in OBJECT_CATEGORIES.items():
        threat_name = THREAT_LEVELS[info['threat']]
        print(f"  {cat_id:2d}. {info['name']:12s} - {threat_name:8s} - {info['description']}")
    
    print("\nObject Categories: PASSED")

def test_model_factory():
    """Test model factory function"""
    print_header("Testing Model Factory")
    
    print("Available models:", list_models())
    
    # Test getting each model
    models = {
        'unet_standard': {'n_channels': 3, 'n_classes': 3},
        'unet_light': {'n_channels': 3, 'n_classes': 3},
        'unet_attention': {'n_channels': 3, 'n_classes': 3},
        'classifier': {'num_classes': 15}
    }
    
    for model_name, kwargs in models.items():
        model = get_model(model_name, **kwargs)
        info = get_model_info(model_name)
        print(f"\n{info['name']}:")
        print(f"  Description: {info['description']}")
        print(f"  Parameters: {info['parameters']}")
        print(f"  Use case: {info['use_case']}")
    
    print("\nModel Factory: PASSED")

def test_checkpoint_save_load():
    """Test checkpoint save and load"""
    print_header("Testing Checkpoint Save/Load")
    
    # Create model
    model = UNetStandard()
    
    # Save checkpoint
    checkpoint_path = "test_checkpoint.pth"
    model.save_checkpoint(
        checkpoint_path,
        epoch=10,
        loss=0.05,
        metrics={'psnr': 28.5, 'ssim': 0.92}
    )
    print(f"Checkpoint saved to {checkpoint_path}")
    
    # Load checkpoint
    model2 = UNetStandard()
    checkpoint = model2.load_checkpoint(checkpoint_path)
    
    print(f"Loaded epoch: {checkpoint['epoch']}")
    print(f"Loaded loss: {checkpoint['loss']:.6f}")
    print(f"Loaded metrics: {checkpoint['metrics']}")
    
    # Cleanup
    import os
    os.remove(checkpoint_path)
    print("Checkpoint file removed")
    
    print("Checkpoint Save/Load: PASSED")

def test_performance():
    """Test inference performance"""
    print_header("Testing Inference Performance")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping performance test")
        return
    
    models_to_test = [
        ('UNet Light', UNetLight()),
        ('UNet Standard', UNetStandard()),
        ('UNet Attention', UNetAttention())
    ]
    
    device = torch.device('cuda')
    input_tensor = torch.randn(1, 3, 256, 256).to(device)
    
    print(f"\nTesting on: {torch.cuda.get_device_name(0)}")
    
    for name, model in models_to_test:
        model = model.to(device)
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(input_tensor)
        
        # Benchmark
        torch.cuda.synchronize()
        import time
        start = time.time()
        
        with torch.no_grad():
            for _ in range(100):
                _ = model(input_tensor)
        
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        fps = 100 / elapsed
        latency = elapsed / 100 * 1000
        
        print(f"\n{name}:")
        print(f"  FPS: {fps:.1f}")
        print(f"  Latency: {latency:.2f}ms")
    
    print("\nPerformance Test: COMPLETED")

def run_all_tests():
    """Run all tests"""
    print("="*60)
    print("  BlueDepth-Crescent Model Testing")
    print("  Maritime Security & Reconnaissance System")
    print("="*60)
    
    tests = [
        ("Base Functionality", test_base_functionality),
        ("UNet Standard", test_unet_standard),
        ("UNet Light", test_unet_light),
        ("UNet Attention", test_unet_attention),
        ("Classifier", test_classifier),
        ("Object Categories", test_object_categories),
        ("Model Factory", test_model_factory),
        ("Checkpoint Save/Load", test_checkpoint_save_load),
        ("Performance", test_performance)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n{test_name}: FAILED")
            print(f"Error: {e}")
            failed += 1
    
    # Summary
    print_header("Test Summary")
    print(f"Total tests: {len(tests)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("\nALL TESTS PASSED!")
        return True
    else:
        print(f"\n{failed} TEST(S) FAILED!")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)