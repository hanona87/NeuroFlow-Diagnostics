"""
PINN Smoke Test: Quick validation of physics-informed neural network.

This script:
1. Initializes PINN and Navier-Stokes residual calculator
2. Generates synthetic spatio-temporal points (x, y, z, t)
3. Performs forward pass through PINN
4. Computes Navier-Stokes residuals (continuity, momentum)
5. Logs individual residual terms
6. Performs one gradient step of residual-based training
7. Verifies shapes and convergence tendency

Output:
  experiments/T3_pinn_smoke/
  ├── residual_history.json
  └── model_checkpoint.pt
"""

import sys
import os
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np

from models.pinn import PhysicsInformedNN, NavierStokesResidualCalculator, HemodynamicCalculator
from utils import set_random_seed, get_device


def run_pinn_smoke_test(
    n_points=50,
    n_steps=20,
    learning_rate=1e-3,
    experiment_dir='experiments/T3_pinn_smoke'
):
    """Run PINN smoke test with physics residual training."""
    
    print("\n" + "="*80)
    print("  PINN SMOKE TEST: Physics-Informed Neural Network")
    print("="*80)
    
    # Setup
    set_random_seed(42)
    device = get_device('cuda')
    experiment_dir = Path(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize components
    print("\n[1/4] Initializing PINN and physics calculator...")
    
    pinn = PhysicsInformedNN(
        input_dim=4,          # (x, y, z, t)
        hidden_layers=[64, 64, 64],
        output_dim=4,         # (u, v, w, p)
        activation='tanh',
        use_fourier_features=False
    ).to(device)
    
    residual_calc = NavierStokesResidualCalculator(
        density=1050.0,
        dynamic_viscosity=3.85e-3,
        characteristic_length=0.01,
        characteristic_velocity=0.3,
        characteristic_time=0.8
    )
    
    hemo_calc = HemodynamicCalculator()
    
    n_params = sum(p.numel() for p in pinn.parameters())
    print(f"  PINN model: {n_params:,} parameters")
    print(f"  Density: {residual_calc.density} kg/m³")
    print(f"  Dynamic viscosity: {residual_calc.dynamic_viscosity} Pa·s")
    print(f"  Kinematic viscosity: {residual_calc.kinematic_viscosity:.2e} m²/s")
    print(f"  Reynolds number: {residual_calc.Re:.2f}")
    
    # Generate synthetic spatio-temporal points
    print("\n[2/4] Generating synthetic spatio-temporal collocation points...")
    
    # Collocation points: random points in domain
    x = torch.rand(n_points, 1, device=device) * 0.1 - 0.05  # x ∈ [-0.05, 0.05]
    y = torch.rand(n_points, 1, device=device) * 0.1 - 0.05  # y ∈ [-0.05, 0.05]
    z = torch.rand(n_points, 1, device=device) * 0.2 - 0.1   # z ∈ [-0.1, 0.1]
    t = torch.rand(n_points, 1, device=device) * 0.8         # t ∈ [0, 0.8]
    
    collocation_points = torch.cat([x, y, z, t], dim=1)
    collocation_points.requires_grad_(True)
    
    print(f"  Generated {n_points} collocation points")
    print(f"  x range: [{collocation_points[:, 0].min().item():.4f}, {collocation_points[:, 0].max().item():.4f}]")
    print(f"  y range: [{collocation_points[:, 1].min().item():.4f}, {collocation_points[:, 1].max().item():.4f}]")
    print(f"  z range: [{collocation_points[:, 2].min().item():.4f}, {collocation_points[:, 2].max().item():.4f}]")
    print(f"  t range: [{collocation_points[:, 3].min().item():.4f}, {collocation_points[:, 3].max().item():.4f}]")
    
    # Physics residual training loop
    print("\n[3/4] Training PINN with physics residuals...")
    
    optimizer = torch.optim.Adam(pinn.parameters(), lr=learning_rate)
    residual_history = []
    
    pinn.train()
    
    for step in range(n_steps):
        # Forward pass
        u_pred = pinn(collocation_points)  # (n_points, 4)
        
        # Compute residuals
        residuals = residual_calc.compute_residuals(
            u_pred, u_pred[:, 3:4], collocation_points, compute_terms_separately=True
        )
        
        # Physics loss: sum of residual magnitudes
        loss = (
            residuals['continuity_mag'] +
            residuals['momentum_x_mag'] +
            residuals['momentum_y_mag'] +
            residuals['momentum_z_mag']
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Log
        residual_history.append({
            'step': step,
            'total_loss': loss.item(),
            'continuity_mag': residuals['continuity_mag'].item(),
            'momentum_x_mag': residuals['momentum_x_mag'].item(),
            'momentum_y_mag': residuals['momentum_y_mag'].item(),
            'momentum_z_mag': residuals['momentum_z_mag'].item(),
            'total_residual': residuals['total'].item()
        })
        
        if (step + 1) % 5 == 0:
            print(f"  Step {step+1:3d}/{n_steps}: loss={loss.item():.6e}")
            print(f"    Continuity: {residuals['continuity_mag'].item():.6e}")
            print(f"    Momentum X: {residuals['momentum_x_mag'].item():.6e}")
            print(f"    Momentum Y: {residuals['momentum_y_mag'].item():.6e}")
            print(f"    Momentum Z: {residuals['momentum_z_mag'].item():.6e}")
    
    # Test hemodynamic calculations
    print("\n[4/4] Testing hemodynamic calculations...")
    
    pinn.eval()
    with torch.no_grad():
        u_final = pinn(collocation_points)
        
        # Generate time series for hemodynamic indices
        n_time = 10
        wss_series = []
        osi_series = []
        
        for t_idx in range(n_time):
            t_val = t_idx / n_time
            t_pts = torch.full((n_points, 1), t_val, device=device)
            pts_t = torch.cat([collocation_points[:, :3], t_pts], dim=1)
            
            u_t = pinn(pts_t)
            
            # Simple WSS approximation from velocity field
            wss_approx = torch.norm(u_t[:, :3], dim=1, keepdim=True) * 0.001
            wss_series.append(wss_approx)
        
        wss_series = torch.stack(wss_series, dim=0).squeeze()  # (n_time, n_points)
        
        # Compute TAWSS
        tawss = HemodynamicCalculator.compute_time_averaged_wss(wss_series)
        print(f"  TAWSS shape: {tawss.shape}")
        print(f"  TAWSS range: [{tawss.min().item():.6e}, {tawss.max().item():.6e}]")
        
        # Generate WSS vector time series for OSI
        wss_vec_series = torch.randn(n_time, n_points, 3, device=device) * 0.001
        osi = HemodynamicCalculator.compute_oscillatory_shear_index(wss_vec_series)
        print(f"  OSI shape: {osi.shape}")
        print(f"  OSI range: [{osi.min().item():.4f}, {osi.max().item():.4f}]")
        
        # Compute RRT
        rrt = HemodynamicCalculator.compute_relative_residence_time(tawss, osi)
        print(f"  RRT shape: {rrt.shape}")
        print(f"  RRT range: [{rrt.min().item():.6e}, {rrt.max().item():.6e}]")
    
    # Save results
    print("\n[SAVE] Saving results...")
    
    with open(experiment_dir / 'residual_history.json', 'w') as f:
        json.dump(residual_history, f, indent=2)
    
    torch.save(pinn.state_dict(), experiment_dir / 'model_checkpoint.pt')
    
    print(f"✅ PINN smoke test complete. Results saved to {experiment_dir}")
    
    # Summary
    print("\n" + "="*80)
    print("  SUMMARY")
    print("="*80)
    print(f"  Initial loss: {residual_history[0]['total_loss']:.6e}")
    print(f"  Final loss:   {residual_history[-1]['total_loss']:.6e}")
    
    loss_reduction = (residual_history[0]['total_loss'] - residual_history[-1]['total_loss']) / residual_history[0]['total_loss']
    print(f"  Loss reduction: {loss_reduction:.2%}")
    
    if loss_reduction > 0:
        print("  ✅ Residuals decreasing (training working)")
    else:
        print("  ⚠️  Residuals not decreasing (may need hyperparameter tuning)")
    
    print("="*80 + "\n")


if __name__ == '__main__':
    run_pinn_smoke_test(
        n_points=50,
        n_steps=20,
        learning_rate=1e-3
    )
