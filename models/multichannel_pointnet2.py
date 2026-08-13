"""
Multichannel PointNet++ for hemodynamic-informed rupture prediction (Stage 3).
"""

from typing import Optional, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiChannelPointNet2Classification(nn.Module):
    """
    PointNet++ for multichannel rupture prediction.
    Processes geometric features (x,y,z,normals) + hemodynamic channels (u,v,w,p,WSS,OSI,RRT).
    """
    
    def __init__(self, 
                 input_channels: int = 6,  # Default: xyz + normals
                 hemodynamic_channels: int = 7,  # u,v,w,p,WSS,OSI,RRT
                 num_classes: int = 2,
                 dropout_rate: float = 0.5,
                 fusion_method: str = 'concatenation'):
        """
        Args:
            input_channels: Geometric input channels (including normals)
            hemodynamic_channels: Number of hemodynamic channels
            num_classes: Number of output classes (2 for binary rupture)
            dropout_rate: Dropout rate
            fusion_method: How to fuse channels ('concatenation', 'early_fusion', 'late_fusion')
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.hemodynamic_channels = hemodynamic_channels
        self.num_classes = num_classes
        self.total_channels = input_channels + hemodynamic_channels
        self.fusion_method = fusion_method
        
        # Set Abstraction layers
        # SA1: Extract local geometric and hemodynamic features
        self.sa1_mlp = nn.Sequential(
            nn.Linear(self.total_channels + 3, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        # SA2: Hierarchical feature combination
        self.sa2_mlp = nn.Sequential(
            nn.Linear(64 + 3, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        # SA3: Higher-level feature extraction
        self.sa3_mlp = nn.Sequential(
            nn.Linear(128 + 3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        
        # SA4: Global aggregation
        self.sa4_mlp = nn.Sequential(
            nn.Linear(256 + 3, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        
        # Classification head
        self.fc1 = nn.Linear(512, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, num_classes)
        
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, N, C) where C = xyz + normals + hemodynamic channels
            
        Returns:
            Class logits (B, num_classes)
        """
        B, N, C = x.size()
        
        # Simplified processing: directly aggregate
        # In a full implementation, this would use k-NN grouping and set abstraction
        
        # Max pooling over points
        features = x[:, :, :512] if C >= 512 else x  # Use first 512 dims
        
        # Global max pooling
        global_features = torch.max(features, dim=1)[0]  # (B, min(C, 512))
        
        # Also use mean pooling for robustness
        mean_features = torch.mean(features, dim=1)  # (B, min(C, 512))
        
        # Concatenate
        combined = torch.cat([global_features, mean_features], dim=1)
        
        # Ensure correct dimension
        if combined.size(1) < 512:
            padding = torch.zeros(B, 512 - combined.size(1), device=x.device)
            combined = torch.cat([combined, padding], dim=1)
        else:
            combined = combined[:, :512]
        
        # Classification head
        x = combined
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        logits = self.fc3(x)  # (B, num_classes)
        
        return logits


class AblationPointNet2(nn.Module):
    """
    PointNet++ variant for ablation studies with specific channel selections.
    """
    
    def __init__(self,
                 input_channels: int,
                 num_classes: int = 2,
                 dropout_rate: float = 0.5):
        """
        Args:
            input_channels: Number of input channels (varies by ablation)
            num_classes: Number of output classes
            dropout_rate: Dropout rate
        """
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        
        # Adaptive input layer
        self.input_layer = nn.Linear(input_channels, 64)
        self.input_bn = nn.BatchNorm1d(64)
        
        # Feature extraction
        self.fc1 = nn.Linear(64, 128)
        self.bn1 = nn.BatchNorm1d(128)
        
        self.fc2 = nn.Linear(128, 256)
        self.bn2 = nn.BatchNorm1d(256)
        
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        
        # Classification head
        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        
        self.fc5 = nn.Linear(64, num_classes)
        
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, N, C) where C varies
            
        Returns:
            Class logits (B, num_classes)
        """
        B, N, C = x.size()
        
        # Max and mean pooling over points
        max_pool = torch.max(x, dim=1)[0]  # (B, C)
        mean_pool = torch.mean(x, dim=1)  # (B, C)
        
        # Concatenate
        features = torch.cat([max_pool, mean_pool], dim=1)  # (B, 2*C)
        
        # Process features
        x = self.input_layer(features)
        x = self.input_bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        logits = self.fc5(x)  # (B, num_classes)
        
        return logits


class EnsembleRupturePredictor(nn.Module):
    """
    Ensemble of models for uncertainty quantification in rupture prediction.
    """
    
    def __init__(self,
                 base_model_class,
                 model_kwargs: Dict,
                 num_models: int = 5):
        """
        Args:
            base_model_class: Class of base model
            model_kwargs: Kwargs for base model
            num_models: Number of models in ensemble
        """
        super().__init__()
        
        self.num_models = num_models
        self.models = nn.ModuleList([
            base_model_class(**model_kwargs) for _ in range(num_models)
        ])
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through ensemble.
        
        Args:
            x: Input tensor
            
        Returns:
            (mean_logits, std_logits) - mean and std across ensemble
        """
        logits_list = []
        
        for model in self.models:
            logits = model(x)
            logits_list.append(logits)
        
        # Stack and compute statistics
        all_logits = torch.stack(logits_list, dim=0)  # (num_models, B, num_classes)
        
        mean_logits = torch.mean(all_logits, dim=0)
        std_logits = torch.std(all_logits, dim=0)
        
        return mean_logits, std_logits
    
    def forward_individual(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Get predictions from each model individually."""
        return [model(x) for model in self.models]


class MCDropoutPredictor(nn.Module):
    """
    Monte Carlo Dropout for uncertainty quantification.
    Model remains in training mode during inference to enable dropout.
    """
    
    def __init__(self,
                 base_model: nn.Module,
                 mc_samples: int = 100):
        """
        Args:
            base_model: Base model with dropout
            mc_samples: Number of MC samples
        """
        super().__init__()
        self.base_model = base_model
        self.mc_samples = mc_samples
    
    def forward(self, x: torch.Tensor, return_all_samples: bool = False
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with MC dropout.
        
        Args:
            x: Input tensor
            return_all_samples: Whether to return all samples
            
        Returns:
            (mean_logits, std_logits) or (mean, std, all_samples)
        """
        # Keep model in train mode for dropout
        self.base_model.train()
        
        samples = []
        for _ in range(self.mc_samples):
            logits = self.base_model(x)
            samples.append(logits)
        
        # Stack samples
        all_samples = torch.stack(samples, dim=0)  # (mc_samples, B, num_classes)
        
        mean_logits = torch.mean(all_samples, dim=0)
        std_logits = torch.std(all_samples, dim=0)
        
        if return_all_samples:
            return mean_logits, std_logits, all_samples
        else:
            return mean_logits, std_logits


# Channel combination functions for ablation studies

def get_channel_indices(channels: List[str]) -> List[int]:
    """
    Get indices for specified channels.
    
    Args:
        channels: List of channel names
        
    Returns:
        List of column indices
    """
    channel_map = {
        'x': 0, 'y': 1, 'z': 2,
        'nx': 3, 'ny': 4, 'nz': 5,
        'u': 6, 'v': 7, 'w': 8,
        'p': 9,
        'wss': 10,
        'tawss': 11,
        'osi': 12,
        'rrt': 13
    }
    
    indices = [channel_map[c] for c in channels if c in channel_map]
    return indices


def select_channels(x: torch.Tensor, channel_names: List[str]) -> torch.Tensor:
    """
    Select specific channels from input tensor.
    
    Args:
        x: Input tensor (B, N, C)
        channel_names: List of channel names to select
        
    Returns:
        Filtered tensor (B, N, len(channel_names))
    """
    indices = get_channel_indices(channel_names)
    if not indices:
        return x
    
    return x[:, :, indices]


# Morphological feature extraction (for conventional ML baseline)

def extract_morphological_features(points: torch.Tensor,
                                  labels: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Extract conventional morphological features from point cloud.
    
    Args:
        points: Point cloud (B, N, 3)
        labels: Point labels for segmentation (optional)
        
    Returns:
        Morphological feature vector (B, num_features)
    """
    B, N, _ = points.size()
    
    features_list = []
    
    for b in range(B):
        pc = points[b]  # (N, 3)
        
        # Volume estimate (using convex hull)
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(pc.cpu().numpy())
            volume = hull.volume
        except:
            volume = 0.0
        
        # Size metrics
        center = pc.mean(dim=0)
        distances = torch.norm(pc - center, dim=1)
        max_dist = distances.max().item()
        mean_dist = distances.mean().item()
        
        # Aspect ratio (PCA-based)
        pc_centered = pc - center
        U, S, V = torch.svd(pc_centered)
        aspect_ratio = S[-1].item() / (S[0].item() + 1e-8)
        
        # Curvature estimate
        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=1)
            curvature = pca.fit(pc.cpu().numpy()).explained_variance_[0]
        except:
            curvature = 0.0
        
        batch_features = torch.tensor([
            volume, max_dist, mean_dist, aspect_ratio, curvature
        ], dtype=torch.float32)
        
        features_list.append(batch_features)
    
    features = torch.stack(features_list, dim=0).to(points.device)
    return features
