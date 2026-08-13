"""
Comprehensive test suite for NeuroFlow-Diagnostics project.
Tests all major components and functionalities.
"""

import sys
import os
import tempfile
import torch
import numpy as np
from pathlib import Path

def test_imports():
    """Test all module imports."""
    print("\n" + "="*60)
    print("TEST 1: Module Imports")
    print("="*60)
    
    try:
        from utils import set_random_seed, get_device, load_config
        print("✅ utils module imported")
        
        from models import PointNet2Classification, PhysicsInformedNN, MultiChannelPointNet2Classification
        print("✅ models module imported")
        
        from losses import FocalLoss, WeightedCrossEntropyLoss, PhysicsLoss
        print("✅ losses module imported")
        
        from evaluation import ClassificationMetrics, CalibrationMetrics
        print("✅ evaluation module imported")
        
        from trainers import BaseTrainer, DetectionTrainer
        print("✅ trainers module imported")
        
        from data.preprocessing import PointCloudPreprocessor, PointCloudDataset, create_synthetic_dataset
        print("✅ data.preprocessing module imported")
        
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_loading():
    """Test configuration loading."""
    print("\n" + "="*60)
    print("TEST 2: Configuration Loading")
    print("="*60)
    
    try:
        from utils import load_config
        
        config = load_config('configs/config.yaml')
        print(f"✅ Config loaded successfully")
        print(f"   - Config keys: {list(config.keys())[:5]}...")
        return True
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        return False


def test_model_instantiation():
    """Test model instantiation."""
    print("\n" + "="*60)
    print("TEST 3: Model Instantiation")
    print("="*60)
    
    try:
        from models import PointNet2Classification, PhysicsInformedNN, MultiChannelPointNet2Classification
        from utils import get_device
        
        device = get_device()
        
        # Test PointNet2Classification
        model1 = PointNet2Classification(in_channels=6, num_classes=2).to(device)
        print(f"✅ PointNet2Classification instantiated ({sum(p.numel() for p in model1.parameters())} params)")
        
        # Test PhysicsInformedNN
        model2 = PhysicsInformedNN(input_dim=4, hidden_layers=[64, 64], output_dim=4).to(device)
        print(f"✅ PhysicsInformedNN instantiated ({sum(p.numel() for p in model2.parameters())} params)")
        
        # Test MultiChannelPointNet2Classification
        model3 = MultiChannelPointNet2Classification(input_channels=10, num_classes=2).to(device)
        print(f"✅ MultiChannelPointNet2Classification instantiated ({sum(p.numel() for p in model3.parameters())} params)")
        
        return True
    except Exception as e:
        print(f"❌ Model instantiation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_loss_functions():
    """Test loss function instantiation and computation."""
    print("\n" + "="*60)
    print("TEST 4: Loss Functions")
    print("="*60)
    
    try:
        from losses import FocalLoss, WeightedCrossEntropyLoss, PhysicsLoss
        import torch
        
        # Test FocalLoss
        focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        preds = torch.randn(4, 2)  # batch_size=4, num_classes=2
        targets = torch.tensor([0, 1, 0, 1])
        loss = focal_loss(preds, targets)
        print(f"✅ FocalLoss computed: {loss.item():.4f}")
        
        # Test WeightedCrossEntropyLoss
        wce_loss = WeightedCrossEntropyLoss(pos_weight=1.0)
        print(f"✅ WeightedCrossEntropyLoss instantiated")
        
        # Test PhysicsLoss
        physics_loss = PhysicsLoss()
        residuals = {
            'continuity': torch.randn(4, 1),
            'momentum': torch.randn(4, 3)
        }
        loss = physics_loss(residuals)
        print(f"✅ PhysicsLoss computed: {loss.item():.4f}")
        
        return True
    except Exception as e:
        print(f"❌ Loss function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_synthetic_data_creation():
    """Test synthetic dataset creation."""
    print("\n" + "="*60)
    print("TEST 5: Synthetic Data Creation")
    print("="*60)
    
    try:
        from data.preprocessing import create_synthetic_dataset
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = create_synthetic_dataset(n_samples=3, output_dir=tmpdir)
            print(f"✅ Synthetic dataset created at: {dataset_path}")
            
            # Verify files exist
            path_obj = Path(dataset_path)
            if path_obj.exists():
                num_files = len(list(path_obj.glob('*')))
                print(f"   - Number of files: {num_files}")
                return True
            else:
                print(f"❌ Dataset path does not exist: {dataset_path}")
                return False
                
    except Exception as e:
        print(f"❌ Synthetic data creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_metrics():
    """Test metrics computation."""
    print("\n" + "="*60)
    print("TEST 6: Metrics Computation")
    print("="*60)
    
    try:
        from evaluation import ClassificationMetrics
        import numpy as np
        
        # Generate mock predictions and targets
        preds = np.array([0, 1, 1, 0, 1, 0, 1, 1])
        targets = np.array([0, 1, 0, 0, 1, 1, 1, 0])
        
        metrics = ClassificationMetrics.compute_metrics(preds, targets, threshold=0.5)
        
        print(f"✅ Metrics computed")
        print(f"   - Metrics: {list(metrics.keys())[:3]}...")
        
        return True
    except Exception as e:
        print(f"❌ Metrics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_device_handling():
    """Test device detection and handling."""
    print("\n" + "="*60)
    print("TEST 7: Device Handling")
    print("="*60)
    
    try:
        from utils import get_device
        
        device = get_device()
        print(f"✅ Device detected: {device}")
        
        # Test tensor creation on device
        tensor = torch.randn(2, 3).to(device)
        print(f"✅ Tensor created on device: shape {tensor.shape}, device {tensor.device}")
        
        return True
    except Exception as e:
        print(f"❌ Device handling test failed: {e}")
        return False


def test_random_seed():
    """Test random seed reproducibility."""
    print("\n" + "="*60)
    print("TEST 8: Random Seed Reproducibility")
    print("="*60)
    
    try:
        from utils import set_random_seed
        import numpy as np
        
        set_random_seed(42)
        val1 = np.random.random()
        val1_t = torch.randn(1).item()
        
        set_random_seed(42)
        val2 = np.random.random()
        val2_t = torch.randn(1).item()
        
        if abs(val1 - val2) < 1e-6 and abs(val1_t - val2_t) < 1e-6:
            print(f"✅ Random seed reproducibility verified")
            return True
        else:
            print(f"❌ Random seed reproducibility failed")
            return False
    except Exception as e:
        print(f"❌ Random seed test failed: {e}")
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "█"*60)
    print("NEUROFLOW-DIAGNOSTICS COMPREHENSIVE TEST SUITE")
    print("█"*60)
    
    tests = [
        ("Module Imports", test_imports),
        ("Configuration", test_config_loading),
        ("Model Instantiation", test_model_instantiation),
        ("Loss Functions", test_loss_functions),
        ("Synthetic Data", test_synthetic_data_creation),
        ("Metrics", test_metrics),
        ("Device Handling", test_device_handling),
        ("Random Seed", test_random_seed),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ Test '{test_name}' crashed: {e}")
            results[test_name] = False
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status:12} - {test_name}")
    
    print("="*60)
    print(f"Results: {passed}/{total} tests passed")
    print("="*60)
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
