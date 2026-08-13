"""
Test physics residuals for Navier-Stokes PINN.
Verifies that residual computation runs and produces expected shapes/magnitudes.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from models.pinn import PhysicsInformedNN, NavierStokesResidualCalculator


def test_pinn_forward_pass():
    """Test PINN forward pass on dummy input."""
    print("\n" + "="*60)
    print("TEST: PINN Forward Pass")
    print("="*60)
    
    try:
        pinn = PhysicsInformedNN(
            input_dim=4,
            hidden_layers=[64, 64],
            output_dim=4,
            activation='tanh'
        )
        
        # Dummy input: (batch_size, 4) = (x, y, z, t)
        x_dummy = torch.randn(10, 4, requires_grad=True)
        
        output = pinn(x_dummy)
        
        print(f"  Input shape: {x_dummy.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Output channels (u, v, w, p): {output.shape[-1]}")
        
        if output.shape == (10, 4):
            print("✅ PASSED: PINN forward pass produces correct output shape")
            return True
        else:
            print("❌ FAILED: Output shape mismatch")
            return False
            
    except Exception as e:
        print(f"❌ EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_residual_computation():
    """Test that Navier-Stokes residuals compute correctly."""
    print("\n" + "="*60)
    print("TEST: Navier-Stokes Residual Computation")
    print("="*60)
    
    try:
        # Initialize calculator
        calc = NavierStokesResidualCalculator(
            density=1050.0,
            dynamic_viscosity=3.85e-3
        )
        
        # Test inputs
        batch_size = 5
        x = torch.randn(batch_size, 4, requires_grad=True)  # (x, y, z, t)
        
        # Forward pass through dummy PINN
        pinn = PhysicsInformedNN(input_dim=4, hidden_layers=[32], output_dim=4)
        u_pred = pinn(x)  # (batch_size, 4) = [u, v, w, p]
        
        # Compute residuals
        residuals = calc.compute_residuals(u_pred, u_pred[:, 3:4], x, compute_terms_separately=True)
        
        print(f"  Input shape: {x.shape}")
        print(f"  PINN output shape: {u_pred.shape}")
        print(f"  Residual keys: {list(residuals.keys())}")
        
        # Check that all required keys are present
        required_keys = ['continuity', 'momentum_x', 'momentum_y', 'momentum_z', 'total',
                        'continuity_mag', 'momentum_x_mag', 'momentum_y_mag', 'momentum_z_mag',
                        'continuity_rms', 'momentum_rms']
        
        missing_keys = [k for k in required_keys if k not in residuals]
        
        if missing_keys:
            print(f"❌ FAILED: Missing keys: {missing_keys}")
            return False
        
        # Check tensor properties
        for key in ['continuity', 'momentum_x', 'momentum_y', 'momentum_z']:
            res = residuals[key]
            print(f"  {key} shape: {res.shape}, mean mag: {res.mean().item():.6e}")
            
            if res.shape[0] != batch_size:
                print(f"❌ FAILED: {key} has wrong batch size")
                return False
        
        print("✅ PASSED: Residual computation successful")
        return True
        
    except Exception as e:
        print(f"❌ EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_residual_magnitudes():
    """Test that residual magnitudes are reasonable."""
    print("\n" + "="*60)
    print("TEST: Residual Magnitude Sanity Check")
    print("="*60)
    
    try:
        calc = NavierStokesResidualCalculator(
            density=1050.0,
            dynamic_viscosity=3.85e-3
        )
        
        x = torch.randn(10, 4, requires_grad=True)
        pinn = PhysicsInformedNN(input_dim=4, hidden_layers=[32, 32], output_dim=4)
        
        with torch.enable_grad():
            u_pred = pinn(x)
            residuals = calc.compute_residuals(u_pred, u_pred[:, 3:4], x)
        
        # Check that total residual is positive and finite
        total = residuals['total'].item()
        
        print(f"  Total residual: {total:.6e}")
        print(f"  Continuity mag: {residuals['continuity_mag'].item():.6e}")
        print(f"  Momentum_x mag: {residuals['momentum_x_mag'].item():.6e}")
        print(f"  Momentum_y mag: {residuals['momentum_y_mag'].item():.6e}")
        print(f"  Momentum_z mag: {residuals['momentum_z_mag'].item():.6e}")
        
        if total > 0 and np.isfinite(total):
            print("✅ PASSED: Residual magnitudes are reasonable")
            return True
        else:
            print(f"❌ FAILED: Total residual = {total} (invalid)")
            return False
            
    except Exception as e:
        print(f"❌ EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hemodynamic_calculator():
    """Test HemodynamicCalculator methods."""
    print("\n" + "="*60)
    print("TEST: Hemodynamic Calculator")
    print("="*60)
    
    try:
        from models.pinn import HemodynamicCalculator
        
        # Dummy WSS and velocity data
        n_points = 20
        n_time = 10
        
        # Test TAWSS
        wss_time = torch.randn(n_time, n_points) * 0.1
        tawss = HemodynamicCalculator.compute_time_averaged_wss(wss_time)
        
        print(f"  WSS time series shape: {wss_time.shape}")
        print(f"  TAWSS shape: {tawss.shape}")
        print(f"  TAWSS mean: {tawss.mean().item():.6e}")
        
        if tawss.shape != (n_points, 1):
            print("❌ FAILED: TAWSS shape mismatch")
            return False
        
        # Test OSI
        wss_vec_time = torch.randn(n_time, n_points, 3) * 0.1
        osi = HemodynamicCalculator.compute_oscillatory_shear_index(wss_vec_time)
        
        print(f"  WSS vector time series shape: {wss_vec_time.shape}")
        print(f"  OSI shape: {osi.shape}")
        print(f"  OSI range: [{osi.min().item():.4f}, {osi.max().item():.4f}]")
        
        if osi.shape != (n_points, 1):
            print("❌ FAILED: OSI shape mismatch")
            return False
        
        # OSI should be in [0, 0.5]
        if not ((osi >= 0).all() and (osi <= 0.5).all()):
            print("❌ FAILED: OSI values outside valid range [0, 0.5]")
            return False
        
        # Test RRT
        rrt = HemodynamicCalculator.compute_relative_residence_time(tawss, osi)
        
        print(f"  RRT shape: {rrt.shape}")
        print(f"  RRT mean: {rrt.mean().item():.6e}")
        
        if rrt.shape != (n_points, 1):
            print("❌ FAILED: RRT shape mismatch")
            return False
        
        print("✅ PASSED: Hemodynamic calculator working correctly")
        return True
        
    except Exception as e:
        print(f"❌ EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    tests = [
        test_pinn_forward_pass,
        test_residual_computation,
        test_residual_magnitudes,
        test_hemodynamic_calculator
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"\n❌ EXCEPTION in {test_func.__name__}: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    print("\n" + "="*60)
    print(f"PHYSICS RESIDUAL TESTS: {sum(results)}/{len(results)} passed")
    print("="*60)
    
    sys.exit(0 if all(results) else 1)
