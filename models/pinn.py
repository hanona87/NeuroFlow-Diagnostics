"""
Physics-Informed Neural Network (PINN) for solving incompressible Navier-Stokes.
"""

from typing import Dict, Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierFeatureEmbedding(nn.Module):
    """Fourier feature embedding for capturing high-frequency variations."""
    
    def __init__(self, in_features: int, num_freqs: int = 10, scale: float = 1.0):
        """
        Args:
            in_features: Input dimension
            num_freqs: Number of frequency bands
            scale: Frequency scale factor
        """
        super().__init__()
        self.in_features = in_features
        self.num_freqs = num_freqs
        self.scale = scale
        
        # Frequency bands (B_k) sampled from Gaussian
        self.register_buffer(
            'B',
            torch.randn(in_features, num_freqs) * scale
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Fourier feature embedding.
        
        Args:
            x: Input features (B, in_features)
            
        Returns:
            Embedded features (B, 2*in_features*num_freqs)
        """
        # Project to frequency space
        proj = torch.matmul(x, self.B)  # (B, num_freqs)
        
        # Apply sin and cos
        sin_features = torch.sin(2 * torch.pi * proj)
        cos_features = torch.cos(2 * torch.pi * proj)
        
        # Concatenate
        embedded = torch.cat([sin_features, cos_features], dim=-1)
        
        return embedded


class PhysicsInformedNN(nn.Module):
    """
    Physics-Informed Neural Network for Navier-Stokes equations.
    
    Architecture: FC network with tanh activation.
    Input: (x, y, z, t) - spatial coordinates and time
    Output: (u, v, w, p) - velocity components and pressure
    """
    
    def __init__(self, 
                 input_dim: int = 4,
                 hidden_layers: List[int] = None,
                 output_dim: int = 4,
                 activation: str = 'tanh',
                 use_fourier_features: bool = False,
                 fourier_freq_bands: int = 10,
                 dropout_rate: float = 0.0):
        """
        Args:
            input_dim: Input dimension (usually 4 for x,y,z,t)
            hidden_layers: List of hidden layer sizes
            output_dim: Output dimension (usually 4 for u,v,w,p)
            activation: Activation function ('tanh', 'relu', 'sin')
            use_fourier_features: Whether to use Fourier feature embedding
            fourier_freq_bands: Number of Fourier frequency bands
            dropout_rate: Dropout rate (usually 0 for PINNs)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers or [64, 128, 256, 128, 64]
        self.activation_name = activation
        self.use_fourier = use_fourier_features
        self.dropout_rate = dropout_rate
        
        # Fourier embedding (optional)
        if use_fourier_features:
            self.fourier_embed = FourierFeatureEmbedding(
                input_dim, fourier_freq_bands, scale=1.0
            )
            embed_dim = 2 * input_dim * fourier_freq_bands
        else:
            embed_dim = input_dim
        
        # Build network
        layers = []
        prev_dim = embed_dim
        
        for hidden_dim in self.hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if activation.lower() == 'tanh':
                layers.append(nn.Tanh())
            elif activation.lower() == 'relu':
                layers.append(nn.ReLU())
            elif activation.lower() == 'sin':
                layers.append(torch.sin)  # Would need custom wrapper
            
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Select activation function
        if activation.lower() == 'tanh':
            self.activation = torch.tanh
        elif activation.lower() == 'relu':
            self.activation = torch.relu
        elif activation.lower() == 'sin':
            self.activation = torch.sin
        else:
            self.activation = torch.tanh
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through PINN.
        
        Args:
            x: Input coordinates (B, 4) where last col is time
            
        Returns:
            Velocity and pressure fields (B, 4)
        """
        if self.use_fourier:
            x_embed = self.fourier_embed(x)
        else:
            x_embed = x
        
        output = self.network(x_embed)
        return output


class NavierStokesResidualCalculator:
    """
    Calculates residuals for incompressible Navier-Stokes equations.
    """
    
    def __init__(self,
                 density: float = 1050.0,
                 dynamic_viscosity: float = 3.85e-3,
                 domain_bounds: Optional[Dict] = None,
                 non_dimensionalization_enabled: bool = True,
                 characteristic_length: float = 0.01,
                 characteristic_velocity: float = 0.3,
                 characteristic_time: float = 0.8):
        """
        Args:
            density: Fluid density ρ = 1050 kg/m³ (blood)
            dynamic_viscosity: Dynamic viscosity μ = 3.85e-3 Pa·s (blood)
                              Kinematic viscosity ν = μ/ρ = 3.66e-6 m²/s
            domain_bounds: Domain bounds for coordinate ranges
            non_dimensionalization_enabled: Whether to apply non-dimensionalization
            characteristic_length: Characteristic length scale (m)
            characteristic_velocity: Characteristic velocity scale (m/s)
            characteristic_time: Characteristic time scale (s), cycle T = 0.8 s for cardiac flow
        """
        self.density = density
        self.dynamic_viscosity = dynamic_viscosity
        # Compute kinematic viscosity: ν = μ / ρ
        self.kinematic_viscosity = dynamic_viscosity / density
        self.domain_bounds = domain_bounds or {
            'x': [-0.05, 0.05],
            'y': [-0.05, 0.05],
            'z': [-0.1, 0.1],
            't': [0, 0.8]
        }
        
        # Non-dimensionalization
        self.non_dim_enabled = non_dimensionalization_enabled
        self.L_char = characteristic_length
        self.U_char = characteristic_velocity
        self.T_char = characteristic_time
        
        # Compute Reynolds number for diagnostics
        self.Re = (self.U_char * self.L_char) / self.kinematic_viscosity
        self.Strouhal = self.L_char / (self.U_char * self.T_char)
    
    def compute_velocity_gradient(self, u: torch.Tensor, x: torch.Tensor
                                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute velocity gradients (∇u).
        
        Args:
            u: Velocity field (B, 1) - should have requires_grad=True
            x: Coordinates (B, 4)
            
        Returns:
            Gradients (du/dx, du/dy, du/dz)
        """
        grads = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True,
            allow_unused=True
        )[0]
        
        return grads[:, 0], grads[:, 1], grads[:, 2]
    
    def compute_laplacian(self, u: torch.Tensor, x: torch.Tensor
                         ) -> torch.Tensor:
        """
        Compute Laplacian (∇²u).
        
        Args:
            u: Velocity component (B, 1)
            x: Coordinates (B, 4)
            
        Returns:
            Laplacian (B, 1)
        """
        grad_u = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True,
            allow_unused=True
        )[0]
        
        # Compute second derivatives
        grad_x, grad_y, grad_z, grad_t = grad_u[:, 0:1], grad_u[:, 1:2], grad_u[:, 2:3], grad_u[:, 3:4]
        
        laplacian = 0
        for grad_comp in [grad_x, grad_y, grad_z]:
            d2_comp = torch.autograd.grad(
                grad_comp, x,
                grad_outputs=torch.ones_like(grad_comp),
                retain_graph=True,
                create_graph=True,
                allow_unused=True
            )[0]
            laplacian = laplacian + d2_comp[:, 0:1] + d2_comp[:, 1:2] + d2_comp[:, 2:3]
        
        return laplacian
    
    def compute_residuals(self, u_pred: torch.Tensor, 
                         p_pred: torch.Tensor,
                         x: torch.Tensor,
                         compute_terms_separately: bool = False
                         ) -> Dict[str, torch.Tensor]:
        """
        Compute all Navier-Stokes residuals.
        
        Args:
            u_pred: Predicted velocity and pressure (B, 4): [u, v, w, p]
            p_pred: Pressure component (included in u_pred[:, 3])
            x: Coordinates (B, 4) with requires_grad=True
            compute_terms_separately: If True, return individual terms
            
        Returns:
            Dictionary with residual tensors:
                - 'continuity': ∇·u
                - 'momentum_x': ∂u/∂t + (u·∇)u + ∇p/ρ - ν∇²u
                - 'momentum_y': similar
                - 'momentum_z': similar
                - 'total': sum of all residuals
        """
        u = u_pred[:, 0:1]
        v = u_pred[:, 1:2]
        w = u_pred[:, 2:3]
        p = u_pred[:, 3:4]
        
        # Compute first derivatives (velocity components and pressure)
        grad_output = torch.ones_like(u)
        
        # ∇u (first derivatives of u)
        du_dxyz = torch.autograd.grad(
            u, x, grad_outputs=grad_output,
            retain_graph=True, create_graph=True, allow_unused=True
        )[0]
        du_dx, du_dy, du_dz, du_dt = du_dxyz[:, 0:1], du_dxyz[:, 1:2], du_dxyz[:, 2:3], du_dxyz[:, 3:4]
        
        # ∇v
        dv_dxyz = torch.autograd.grad(
            v, x, grad_outputs=grad_output,
            retain_graph=True, create_graph=True, allow_unused=True
        )[0]
        dv_dx, dv_dy, dv_dz, dv_dt = dv_dxyz[:, 0:1], dv_dxyz[:, 1:2], dv_dxyz[:, 2:3], dv_dxyz[:, 3:4]
        
        # ∇w
        dw_dxyz = torch.autograd.grad(
            w, x, grad_outputs=grad_output,
            retain_graph=True, create_graph=True, allow_unused=True
        )[0]
        dw_dx, dw_dy, dw_dz, dw_dt = dw_dxyz[:, 0:1], dw_dxyz[:, 1:2], dw_dxyz[:, 2:3], dw_dxyz[:, 3:4]
        
        # ∇p
        dp_dxyz = torch.autograd.grad(
            p, x, grad_outputs=grad_output,
            retain_graph=True, create_graph=True, allow_unused=True
        )[0]
        dp_dx, dp_dy, dp_dz = dp_dxyz[:, 0:1], dp_dxyz[:, 1:2], dp_dxyz[:, 2:3]
        
        # CONTINUITY EQUATION: ∇·u = ∂u/∂x + ∂v/∂y + ∂w/∂z = 0
        continuity_residual = du_dx + dv_dy + dw_dz
        
        # MOMENTUM EQUATIONS
        # Compute second derivatives (Laplacians)
        d2u_dx2 = torch.autograd.grad(du_dx, x, grad_outputs=torch.ones_like(du_dx),
                                      retain_graph=True, create_graph=True, allow_unused=True)[0][:, 0:1]
        d2u_dy2 = torch.autograd.grad(du_dy, x, grad_outputs=torch.ones_like(du_dy),
                                      retain_graph=True, create_graph=True, allow_unused=True)[0][:, 1:2]
        d2u_dz2 = torch.autograd.grad(du_dz, x, grad_outputs=torch.ones_like(du_dz),
                                      retain_graph=True, create_graph=True, allow_unused=True)[0][:, 2:3]
        laplacian_u = d2u_dx2 + d2u_dy2 + d2u_dz2
        
        d2v_dx2 = torch.autograd.grad(dv_dx, x, grad_outputs=torch.ones_like(dv_dx),
                                      retain_graph=True, create_graph=True, allow_unused=True)[0][:, 0:1]
        d2v_dy2 = torch.autograd.grad(dv_dy, x, grad_outputs=torch.ones_like(dv_dy),
                                      retain_graph=True, create_graph=True, allow_unused=True)[0][:, 1:2]
        d2v_dz2 = torch.autograd.grad(dv_dz, x, grad_outputs=torch.ones_like(dv_dz),
                                      retain_graph=True, create_graph=True, allow_unused=True)[0][:, 2:3]
        laplacian_v = d2v_dx2 + d2v_dy2 + d2v_dz2
        
        d2w_dx2 = torch.autograd.grad(dw_dx, x, grad_outputs=torch.ones_like(dw_dx),
                                      retain_graph=True, create_graph=True, allow_unused=True)[0][:, 0:1]
        d2w_dy2 = torch.autograd.grad(dw_dy, x, grad_outputs=torch.ones_like(dw_dy),
                                      retain_graph=True, create_graph=True, allow_unused=True)[0][:, 1:2]
        d2w_dz2 = torch.autograd.grad(dw_dz, x, grad_outputs=torch.ones_like(dw_dz),
                                      retain_graph=True, create_graph=True, allow_unused=True)[0][:, 2:3]
        laplacian_w = d2w_dx2 + d2w_dy2 + d2w_dz2
        
        # X-momentum: ∂u/∂t + (u·∇)u + ∇p/ρ - ν∇²u = 0
        convection_x = u * du_dx + v * du_dy + w * du_dz
        momentum_x_residual = du_dt + convection_x + dp_dx / self.density - self.kinematic_viscosity * laplacian_u
        
        # Y-momentum: ∂v/∂t + (u·∇)v + ∇p/ρ - ν∇²v = 0
        convection_y = u * dv_dx + v * dv_dy + w * dv_dz
        momentum_y_residual = dv_dt + convection_y + dp_dy / self.density - self.kinematic_viscosity * laplacian_v
        
        # Z-momentum: ∂w/∂t + (u·∇)w + ∇p/ρ - ν∇²w = 0
        convection_z = u * dw_dx + v * dw_dy + w * dw_dz
        momentum_z_residual = dw_dt + convection_z + dp_dz / self.density - self.kinematic_viscosity * laplacian_w
        
        residuals = {
            'continuity': continuity_residual,
            'momentum_x': momentum_x_residual,
            'momentum_y': momentum_y_residual,
            'momentum_z': momentum_z_residual,
        }
        
        # Compute individual term magnitudes for logging
        residuals['continuity_mag'] = torch.mean(torch.abs(continuity_residual))
        residuals['momentum_x_mag'] = torch.mean(torch.abs(momentum_x_residual))
        residuals['momentum_y_mag'] = torch.mean(torch.abs(momentum_y_residual))
        residuals['momentum_z_mag'] = torch.mean(torch.abs(momentum_z_residual))
        
        # Total residual (sum of squared terms)
        total_residual = (
            torch.mean(continuity_residual ** 2) +
            torch.mean(momentum_x_residual ** 2) +
            torch.mean(momentum_y_residual ** 2) +
            torch.mean(momentum_z_residual ** 2)
        )
        
        residuals['total'] = total_residual
        
        if compute_terms_separately:
            residuals['continuity_rms'] = torch.sqrt(torch.mean(continuity_residual ** 2))
            residuals['momentum_rms'] = torch.sqrt(
                (torch.mean(momentum_x_residual ** 2) +
                 torch.mean(momentum_y_residual ** 2) +
                 torch.mean(momentum_z_residual ** 2)) / 3.0
            )
        
        return residuals


class HemodynamicCalculator:
    """
    Compute derived hemodynamic quantities from flow fields.
    
    Reference hemodynamic indices:
    - TAWSS (Time-Averaged Wall Shear Stress): (1/T) ∫_0^T |WSS(t)| dt
    - OSI (Oscillatory Shear Index): 0.5 * (1 - |∫_0^T WSS_vec dt| / ∫_0^T |WSS_vec| dt)
    - RRT (Relative Residence Time): 1 / (TAWSS * (1 - 2*OSI))
    """
    
    @staticmethod
    def compute_wall_shear_stress(velocity_grad_normal: torch.Tensor,
                                  dynamic_viscosity: float = 3.85e-3) -> torch.Tensor:
        """
        Compute wall shear stress magnitude.
        
        WSS = μ * |∂u/∂n| where n is the wall-normal direction
        
        Args:
            velocity_grad_normal: Velocity gradient in normal direction (N, 1)
            dynamic_viscosity: Dynamic viscosity μ = 3.85e-3 Pa·s
            
        Returns:
            WSS magnitude (N, 1)
        """
        # WSS magnitude: μ * |velocity gradient normal to wall|
        wss = dynamic_viscosity * torch.abs(velocity_grad_normal)
        return wss
    
    @staticmethod
    def compute_time_averaged_wss(wss_time_series: torch.Tensor) -> torch.Tensor:
        """
        Compute Time-Averaged Wall Shear Stress (TAWSS).
        
        TAWSS = (1/T) ∫_0^T |WSS(t)| dt
        
        Args:
            wss_time_series: WSS magnitude over time (T, N) where T is time points
            
        Returns:
            TAWSS (N, 1)
        """
        # Average over time dimension
        tawss = torch.mean(torch.abs(wss_time_series), dim=0, keepdim=True)
        return tawss.T  # (N, 1)
    
    @staticmethod
    def compute_oscillatory_shear_index(wss_vec_time_series: torch.Tensor) -> torch.Tensor:
        """
        Compute Oscillatory Shear Index (OSI).
        
        OSI = 0.5 * (1 - |∫_0^T WSS_vec(t) dt| / ∫_0^T |WSS_vec(t)| dt)
        
        Measures the directional oscillation of wall shear stress.
        OSI ≈ 0: unidirectional flow (low risk)
        OSI ≈ 0.5: oscillatory flow (high risk for aneurysm rupture)
        
        Args:
            wss_vec_time_series: WSS vector over time (T, N, 3) where T is time points
            
        Returns:
            OSI (N, 1)
        """
        # Integrate WSS vector over time
        wss_integral_vec = torch.mean(wss_vec_time_series, dim=0)  # (N, 3)
        wss_integral_mag = torch.norm(wss_integral_vec, dim=1, keepdim=True)  # (N, 1)
        
        # Integrate magnitude over time
        wss_mag_time_series = torch.norm(wss_vec_time_series, dim=2)  # (T, N)
        wss_integral_mag_avg = torch.mean(wss_mag_time_series, dim=0, keepdim=True).T  # (N, 1)
        
        # Avoid division by zero
        osi = 0.5 * (1.0 - wss_integral_mag / (wss_integral_mag_avg + 1e-8))
        
        # Clamp to [0, 0.5] (valid range)
        osi = torch.clamp(osi, 0, 0.5)
        
        return osi
    
    @staticmethod
    def compute_relative_residence_time(tawss: torch.Tensor,
                                        osi: torch.Tensor) -> torch.Tensor:
        """
        Compute Relative Residence Time (RRT).
        
        RRT = 1 / (TAWSS * (1 - 2*OSI))
        
        Measures how long blood residence in a region.
        High RRT indicates stagnant flow prone to thrombosis/growth.
        
        Args:
            tawss: Time-Averaged WSS (N, 1)
            osi: Oscillatory Shear Index (N, 1)
            
        Returns:
            RRT (N, 1)
        """
        # Avoid singularities
        denominator = tawss * (1.0 - 2.0 * osi) + 1e-8
        rrt = 1.0 / denominator
        
        return rrt
