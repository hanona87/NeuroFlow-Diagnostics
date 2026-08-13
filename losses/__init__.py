"""Losses subpackage."""

from .losses import (
    WeightedCrossEntropyLoss,
    FocalLoss,
    BinaryFocalLoss,
    PhysicsLoss,
    CalibratedCrossEntropyLoss,
    VariationalLoss,
    MultiTaskLoss,
    compute_class_weights,
    adaptive_loss_weights
)

__all__ = [
    'WeightedCrossEntropyLoss',
    'FocalLoss',
    'BinaryFocalLoss',
    'PhysicsLoss',
    'CalibratedCrossEntropyLoss',
    'VariationalLoss',
    'MultiTaskLoss',
    'compute_class_weights',
    'adaptive_loss_weights'
]
