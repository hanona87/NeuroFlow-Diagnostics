"""
Loss functions for NeuroFlow-Diagnostics.
Includes weighted cross-entropy, focal loss, and physics-informed losses.
"""

from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedCrossEntropyLoss(nn.Module):
    """Weighted cross-entropy loss for handling class imbalance."""
    
    def __init__(self, pos_weight: float = 1.0, reduction: str = 'mean'):
        """
        Args:
            pos_weight: Weight for positive class
            reduction: 'mean' or 'sum'
        """
        super().__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Model predictions (B, num_classes)
            targets: Ground truth labels (B,)
            
        Returns:
            Loss value
        """
        if logits.dim() == 2 and logits.size(1) == 2:
            # Binary classification with 2 classes
            probs = F.softmax(logits, dim=1)
            loss = F.cross_entropy(logits, targets, reduction=self.reduction,
                                  weight=torch.tensor([1.0, self.pos_weight],
                                                     device=logits.device))
        else:
            # Multi-class
            loss = F.cross_entropy(logits, targets, reduction=self.reduction)
        
        return loss


class FocalLoss(nn.Module):
    """Focal loss for hard example mining (Lin et al., 2017)."""
    
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, 
                 reduction: str = 'mean'):
        """
        Args:
            alpha: Weight for positive class
            gamma: Focusing parameter
            reduction: 'mean' or 'sum'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Model predictions (B, num_classes)
            targets: Ground truth labels (B,)
            
        Returns:
            Focal loss value
        """
        # Compute softmax
        probs = F.softmax(logits, dim=1)
        
        # Get class probabilities
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Compute focal loss: -alpha * (1 - p_t)^gamma * log(p_t)
        ce = F.cross_entropy(logits, targets, reduction='none')
        focal_weight = (1 - p_t) ** self.gamma
        loss = self.alpha * focal_weight * ce
        
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        
        return loss


class BinaryFocalLoss(nn.Module):
    """Focal loss for binary classification."""
    
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0,
                 reduction: str = 'mean'):
        """
        Args:
            alpha: Weight for positive class
            gamma: Focusing parameter
            reduction: 'mean' or 'sum'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            probs: Model predictions (B,) - probabilities
            targets: Ground truth labels (B,)
            
        Returns:
            Binary focal loss value
        """
        # Ensure targets are float
        targets = targets.float()
        
        # Compute binary cross-entropy
        bce = F.binary_cross_entropy(probs, targets, reduction='none')
        
        # Compute focal weight
        p_t = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply alpha weighting
        alpha_weight = torch.where(
            targets == 1,
            torch.full_like(targets, self.alpha),
            torch.full_like(targets, 1 - self.alpha)
        )
        
        loss = alpha_weight * focal_weight * bce
        
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        
        return loss


class PhysicsLoss(nn.Module):
    """Loss for physics constraints (Navier-Stokes residuals)."""
    
    def __init__(self, weights: Optional[Dict[str, float]] = None,
                 reduction: str = 'mean'):
        """
        Args:
            weights: Dictionary of weights for different residual terms
            reduction: 'mean' or 'sum'
        """
        super().__init__()
        self.weights = weights or {
            'continuity': 1.0,
            'momentum': 1.0
        }
        self.reduction = reduction
    
    def forward(self, residuals: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            residuals: Dictionary containing 'continuity', 'momentum_x', etc.
            
        Returns:
            Physics loss value
        """
        loss = 0.0
        
        # Continuity loss
        if 'continuity' in residuals:
            continuity_loss = torch.mean(residuals['continuity'] ** 2)
            loss += self.weights.get('continuity', 1.0) * continuity_loss
        
        # Momentum losses
        momentum_loss = 0.0
        for component in ['momentum_x', 'momentum_y', 'momentum_z']:
            if component in residuals:
                momentum_loss += torch.mean(residuals[component] ** 2)
        
        loss += self.weights.get('momentum', 1.0) * (momentum_loss / 3.0)
        
        return loss


class CalibratedCrossEntropyLoss(nn.Module):
    """Cross-entropy loss with calibration term."""
    
    def __init__(self, lambda_calib: float = 0.1, reduction: str = 'mean'):
        """
        Args:
            lambda_calib: Weight for calibration term
            reduction: 'mean' or 'sum'
        """
        super().__init__()
        self.lambda_calib = lambda_calib
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Model predictions (B, num_classes)
            targets: Ground truth labels (B,)
            
        Returns:
            Calibrated cross-entropy loss
        """
        # Standard cross-entropy
        ce_loss = F.cross_entropy(logits, targets, reduction=self.reduction)
        
        # Calibration term: minimize entropy of predictions
        probs = F.softmax(logits, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        calib_loss = torch.mean(entropy)
        
        loss = ce_loss + self.lambda_calib * calib_loss
        
        return loss


class VariationalLoss(nn.Module):
    """Variational loss for uncertainty quantification."""
    
    def __init__(self, lambda_var: float = 0.1, reduction: str = 'mean'):
        """
        Args:
            lambda_var: Weight for variance term
            reduction: 'mean' or 'sum'
        """
        super().__init__()
        self.lambda_var = lambda_var
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Model predictions (B, num_classes)
            targets: Ground truth labels (B,)
            
        Returns:
            Variational loss
        """
        # Standard cross-entropy
        ce_loss = F.cross_entropy(logits, targets, reduction=self.reduction)
        
        # Variance term: encourage high confidence
        probs = F.softmax(logits, dim=1)
        max_probs = torch.max(probs, dim=1)[0]
        var_term = -torch.log(max_probs + 1e-8).mean()
        
        loss = ce_loss + self.lambda_var * var_term
        
        return loss


class MultiTaskLoss(nn.Module):
    """Combined loss for multi-task learning."""
    
    def __init__(self, task_weights: Optional[Dict[str, float]] = None):
        """
        Args:
            task_weights: Dictionary mapping task names to weights
        """
        super().__init__()
        self.task_weights = task_weights or {'classification': 1.0, 'physics': 0.5}
    
    def forward(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            losses: Dictionary of loss components
            
        Returns:
            Combined loss
        """
        total_loss = 0.0
        
        for task_name, loss_value in losses.items():
            weight = self.task_weights.get(task_name, 1.0)
            total_loss += weight * loss_value
        
        return total_loss


def compute_class_weights(labels: torch.Tensor) -> torch.Tensor:
    """
    Compute class weights inversely proportional to class frequencies.
    
    Args:
        labels: Class labels (N,)
        
    Returns:
        Class weights (num_classes,)
    """
    unique_labels, counts = torch.unique(labels, return_counts=True)
    num_classes = unique_labels.max().item() + 1
    
    weights = torch.zeros(num_classes, device=labels.device)
    total_samples = len(labels)
    
    for label, count in zip(unique_labels, counts):
        weights[label] = total_samples / (count * num_classes)
    
    return weights


def adaptive_loss_weights(epoch: int, total_epochs: int,
                         initial_weights: Dict[str, float],
                         target_weights: Dict[str, float]) -> Dict[str, float]:
    """
    Compute adaptive loss weights that change during training.
    
    Args:
        epoch: Current epoch
        total_epochs: Total number of epochs
        initial_weights: Initial weights at start
        target_weights: Target weights at end
        
    Returns:
        Current weights dictionary
    """
    progress = epoch / total_epochs
    weights = {}
    
    for key in initial_weights.keys():
        initial = initial_weights[key]
        target = target_weights[key]
        weights[key] = initial + (target - initial) * progress
    
    return weights
