"""
PointNet++ architecture implementation.
Hierarchical feature learning for 3D point clouds.
"""

from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class FarthestPointSampling(torch.autograd.Function):
    """
    Farthest Point Sampling operation for point clouds.
    """
    
    @staticmethod
    def forward(ctx, xyz: torch.Tensor, npoint: int) -> torch.Tensor:
        """
        Args:
            xyz: Point coordinates (B, N, 3)
            npoint: Number of points to sample
            
        Returns:
            Sampled point indices (B, npoint)
        """
        xyz = xyz.contiguous()
        B, N, _ = xyz.size()
        
        centroids = torch.zeros((B, npoint), dtype=torch.long, device=xyz.device)
        distance = torch.ones((B, N), device=xyz.device) * 1e10
        
        batch_indices = torch.arange(B, device=xyz.device)
        
        # Random initial point
        farthest_idx = torch.randint(0, N, (B,), device=xyz.device)
        
        for i in range(npoint):
            centroids[:, i] = farthest_idx
            
            # Get coordinates of current farthest point
            centroid = xyz[batch_indices, farthest_idx, :].unsqueeze(1)
            
            # Compute distances to all points
            dist = torch.sum((xyz - centroid) ** 2, dim=2)
            
            # Update distance with minimum
            mask = dist < distance
            distance[mask] = dist[mask]
            
            # Find next farthest point
            farthest_idx = torch.argmax(distance, dim=1)
        
        return centroids

    @staticmethod
    def backward(ctx, grad_output):
        return None, None


def fps(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """Farthest Point Sampling."""
    return FarthestPointSampling.apply(xyz, npoint)


def query_ball_point(radius: float, nsample: int, xyz: torch.Tensor, 
                     new_xyz: torch.Tensor) -> torch.Tensor:
    """
    Ball query to find neighborhood points.
    
    Args:
        radius: Query radius
        nsample: Maximum number of samples
        xyz: All point coordinates (B, N, 3)
        new_xyz: Query point coordinates (B, npoint, 3)
        
    Returns:
        Group indices (B, npoint, nsample)
    """
    B, N, _ = xyz.size()
    _, npoint, _ = new_xyz.size()
    
    # Compute distances
    dist = torch.cdist(new_xyz, xyz)  # (B, npoint, N)
    
    # Query ball
    group_idx = torch.argsort(dist, dim=2)[:, :, :nsample]  # (B, npoint, nsample)
    
    # Mask points outside radius
    group_dist = torch.gather(dist, 2, group_idx)
    mask = group_dist > radius
    group_idx[mask] = 0
    
    return group_idx


class SetAbstraction(nn.Module):
    """Set Abstraction layer for PointNet++."""
    
    def __init__(self, npoint: int, radius: float, nsample: int, 
                 in_channels: int, out_channels: int, 
                 msg: bool = True, group_all: bool = False):
        """
        Args:
            npoint: Number of points for FPS
            radius: Radius for ball query
            nsample: Number of samples in local region
            in_channels: Input channels
            out_channels: Output channels
            msg: Use multi-scale grouping
            group_all: If True, group all points (for last layer)
        """
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.msg = msg
        self.group_all = group_all
        
        # MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(in_channels + 3, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, xyz: torch.Tensor, features: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            xyz: Point coordinates (B, N, 3)
            features: Point features (B, N, C) or None
            
        Returns:
            (new_xyz, new_features) where:
                new_xyz: Sampled point coordinates (B, npoint, 3)
                new_features: Sampled features (B, npoint, out_channels)
        """
        B, N, _ = xyz.size()
        
        if self.group_all:
            # Group all points for last layer
            new_xyz = xyz.mean(dim=1, keepdim=True)  # (B, 1, 3)
            group_idx = torch.arange(N, device=xyz.device).unsqueeze(0).unsqueeze(0).expand(B, 1, N)
        else:
            # FPS to select centroids
            centroids_idx = fps(xyz, self.npoint)  # (B, npoint)
            
            # Gather new_xyz by index: use direct indexing instead of complex gather
            batch_idx = torch.arange(B, device=xyz.device).view(B, 1).expand(-1, self.npoint)
            new_xyz = xyz[batch_idx, centroids_idx]  # (B, npoint, 3)
            
            # K-NN to find neighbors (simpler than ball query for this implementation)
            group_idx = torch.topk(
                torch.cdist(new_xyz, xyz),
                k=min(self.nsample, N),
                dim=2,
                largest=False
            )[1]  # (B, npoint, nsample)
        
        # Get grouped points
        grouped_xyz = torch.gather(
            xyz.unsqueeze(1).expand(-1, self.npoint if not self.group_all else 1, -1, -1),
            2,
            group_idx.unsqueeze(-1).expand(-1, -1, -1, 3)
        )  # (B, npoint/1, nsample, 3)
        
        # Normalize coordinates
        grouped_xyz = grouped_xyz - new_xyz.unsqueeze(2)
        
        # Concatenate with features
        if features is not None:
            grouped_features = torch.gather(
                features.unsqueeze(1).expand(-1, new_xyz.size(1), -1, -1),
                2,
                group_idx.unsqueeze(-1).expand(-1, -1, -1, features.size(-1))
            )
            grouped = torch.cat([grouped_xyz, grouped_features], dim=-1)
        else:
            grouped = grouped_xyz
        
        # Apply MLP
        B, npoint, nsample, C = grouped.size()
        grouped = grouped.reshape(B * npoint * nsample, C)
        out = self.mlp(grouped)
        out = out.reshape(B, npoint, nsample, -1)
        
        # Max pooling
        out = out.max(dim=2)[0]  # (B, npoint, out_channels)
        
        return new_xyz, out


class FeaturePropagation(nn.Module):
    """Feature Propagation layer for PointNet++."""
    
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        """
        Args:
            in_channels: Channels from lower layer
            skip_channels: Channels from skip connection
            out_channels: Output channels
        """
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(in_channels + skip_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, xyz: torch.Tensor, xyz_prev: torch.Tensor,
                features: torch.Tensor, features_prev: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        """
        Args:
            xyz: Current layer points (B, N, 3)
            xyz_prev: Previous layer points (B, N_prev, 3)
            features: Current layer features (B, N, C)
            features_prev: Previous layer features (B, N_prev, C_prev) or None
            
        Returns:
            Propagated features (B, N_prev, out_channels)
        """
        # Interpolate using inverse distance weighting
        dist = torch.cdist(xyz_prev, xyz)  # (B, N_prev, N)
        
        # Avoid division by zero
        dist[dist < 1e-8] = 1e-8
        weights = 1.0 / (dist + 1e-8)
        weights = weights / (weights.sum(dim=2, keepdim=True) + 1e-8)
        
        interpolated_features = torch.bmm(weights, features)  # (B, N_prev, C)
        
        # Concatenate with skip connection
        if features_prev is not None:
            combined = torch.cat([interpolated_features, features_prev], dim=-1)
        else:
            # No skip connection at this stage
            combined = interpolated_features
        
        # Apply MLP
        B, N, C = combined.size()
        combined = combined.reshape(B * N, C)
        out = self.mlp(combined)
        out = out.reshape(B, N, -1)
        
        return out


class PointNet2Classification(nn.Module):
    """
    PointNet++ for classification tasks.
    
    Set Abstraction configuration (research spec):
      SA1: 2048 pts, r=0.05 → 64 channels
      SA2: 512 pts, r=0.10 → 128 channels
      SA3: 128 pts, r=0.20 → 256 channels
      SA4: 32 pts, r=0.40 → 512 channels
    """
    
    def __init__(self, in_channels: int = 3, num_classes: int = 2,
                 use_normals: bool = False, dropout_rate: float = 0.5):
        """
        Args:
            in_channels: Input channels (3 for xyz only, 6 if xyz+normals)
                        Note: in_channels is the TOTAL input dim, not just features
            num_classes: Number of classification classes
            use_normals: Whether to include normals
            dropout_rate: Dropout rate
        """
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.use_normals = use_normals
        
        # Compute feature dimension (everything except xyz)
        feature_dim = max(0, in_channels - 3)
        
        # Set Abstraction layers with research-spec point counts and radii
        self.sa1 = SetAbstraction(npoint=2048, radius=0.05, nsample=32,
                                  in_channels=feature_dim, out_channels=64)
        self.sa2 = SetAbstraction(npoint=512, radius=0.1, nsample=32,
                                  in_channels=64, out_channels=128)
        self.sa3 = SetAbstraction(npoint=128, radius=0.2, nsample=32,
                                  in_channels=128, out_channels=256)
        # SA4: final abstraction to 32 points (not group_all)
        self.sa4 = SetAbstraction(npoint=32, radius=0.4, nsample=32,
                                  in_channels=256, out_channels=512,
                                  group_all=False)
        
        # Classification head
        self.fc1 = nn.Linear(512, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, num_classes)
        
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xyz: Input point cloud (B, N, 3+) where first 3 dims are coordinates
            
        Returns:
            Class logits (B, num_classes)
        """
        # Separate xyz coordinates (first 3 dims) from features (remaining dims)
        xyz_coords = xyz[:, :, :3]  # (B, N, 3)
        features = xyz[:, :, 3:] if xyz.size(2) > 3 else None  # (B, N, C) or None
        
        # Forward through set abstraction layers
        xyz1, feat1 = self.sa1(xyz_coords, features)
        xyz2, feat2 = self.sa2(xyz1, feat1)
        xyz3, feat3 = self.sa3(xyz2, feat2)
        xyz4, feat4 = self.sa4(xyz3, feat3)
        
        # Global pooling: feat4 is (B, 32, 512), take max over points
        feat_global = feat4.max(dim=1)[0]  # (B, 512)
        
        # Classification head
        x = feat_global
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


class PointNet2Segmentation(nn.Module):
    """
    PointNet++ for per-point segmentation/localization.
    """
    
    def __init__(self, in_channels: int = 3, num_classes: int = 2,
                 dropout_rate: float = 0.5):
        """
        Args:
            in_channels: Input channels (total, including xyz)
            num_classes: Number of segmentation classes
            dropout_rate: Dropout rate
        """
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # Compute feature dimension (everything except xyz)
        feature_dim = max(0, in_channels - 3)
        
        # Set Abstraction layers
        self.sa1 = SetAbstraction(npoint=2048, radius=0.05, nsample=32,
                                  in_channels=feature_dim, out_channels=64)
        self.sa2 = SetAbstraction(npoint=512, radius=0.1, nsample=32,
                                  in_channels=64, out_channels=128)
        self.sa3 = SetAbstraction(npoint=128, radius=0.2, nsample=32,
                                  in_channels=128, out_channels=256)
        self.sa4 = SetAbstraction(npoint=32, radius=0.4, nsample=32,
                                  in_channels=256, out_channels=512)
        
        # Feature Propagation layers
        self.fp4 = FeaturePropagation(in_channels=512, skip_channels=256,
                                      out_channels=256)
        self.fp3 = FeaturePropagation(in_channels=256, skip_channels=128,
                                      out_channels=128)
        self.fp2 = FeaturePropagation(in_channels=128, skip_channels=64,
                                      out_channels=64)
        self.fp1 = FeaturePropagation(in_channels=64, skip_channels=in_channels,
                                      out_channels=32)
        
        # Segmentation head
        self.fc1 = nn.Linear(32, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, 16)
        self.bn2 = nn.BatchNorm1d(16)
        self.fc3 = nn.Linear(16, num_classes)
        
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xyz: Input point cloud (B, N, 3+) where first 3 dims are coordinates
            
        Returns:
            Per-point class logits (B, N, num_classes)
        """
        # Separate xyz coordinates (first 3 dims) from features (remaining dims)
        xyz_coords = xyz[:, :, :3]  # (B, N, 3)
        features = xyz[:, :, 3:] if xyz.size(2) > 3 else None  # (B, N, C) or None
        
        # Forward through set abstraction
        xyz1, feat1 = self.sa1(xyz_coords, features)
        xyz2, feat2 = self.sa2(xyz1, feat1)
        xyz3, feat3 = self.sa3(xyz2, feat2)
        xyz4, feat4 = self.sa4(xyz3, feat3)
        
        # Backward through feature propagation
        feat3_prop = self.fp4(xyz4, xyz3, feat4, feat3)
        feat2_prop = self.fp3(xyz3, xyz2, feat3_prop, feat2)
        feat1_prop = self.fp2(xyz2, xyz1, feat2_prop, feat1)
        feat0_prop = self.fp1(xyz1, xyz_coords, feat1_prop, features)  # To original points
        
        # Segmentation head
        x = feat0_prop  # (B, N, 32)
        B, N, C = x.size()
        x = x.reshape(B * N, C)
        
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        logits = self.fc3(x)  # (B*N, num_classes)
        logits = logits.reshape(B, N, self.num_classes)
        
        return logits
