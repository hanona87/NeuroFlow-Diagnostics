"""
Training infrastructure for NeuroFlow-Diagnostics.
Includes base trainer, dataset loading, and training utilities.
"""

import os
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import h5py
from tqdm import tqdm

from data.preprocessing.preprocessing import PointCloudDataset
from utils import set_random_seed


class PointCloudDatasetLoader:
    """Load point cloud datasets from HDF5 files."""
    
    @staticmethod
    def load_dataset(h5_path: str, split: str = 'train',
                    augmentation: bool = False,
                    augmentation_config: Optional[Dict] = None) -> PointCloudDataset:
        """
        Load a point cloud dataset.
        
        Args:
            h5_path: Path to HDF5 file
            split: Data split name
            augmentation: Whether to apply augmentation
            augmentation_config: Augmentation config dict
            
        Returns:
            PointCloudDataset instance
        """
        dataset = PointCloudDataset(h5_path, augmentation=augmentation,
                                   augmentation_config=augmentation_config)
        return dataset
    
    @staticmethod
    def get_dataloader(h5_path: str, batch_size: int = 20,
                      shuffle: bool = True, num_workers: int = 4,
                      augmentation: bool = False,
                      augmentation_config: Optional[Dict] = None) -> DataLoader:
        """
        Get a DataLoader for a dataset.
        
        Args:
            h5_path: Path to HDF5 file
            batch_size: Batch size
            shuffle: Whether to shuffle
            num_workers: Number of workers
            augmentation: Whether to apply augmentation
            augmentation_config: Augmentation config dict
            
        Returns:
            DataLoader instance
        """
        dataset = PointCloudDatasetLoader.load_dataset(
            h5_path, augmentation=augmentation,
            augmentation_config=augmentation_config
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return dataloader


class BaseTrainer:
    """Base trainer class for model training."""
    
    def __init__(self, 
                 model: nn.Module,
                 device: torch.device,
                 experiment_dir: str,
                 seed: int = 42):
        """
        Initialize trainer.
        
        Args:
            model: Neural network model
            device: torch.device
            experiment_dir: Directory for experiment outputs
            seed: Random seed
        """
        self.model = model
        self.device = device
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        set_random_seed(seed)
        
        self.model = self.model.to(device)
        
        # Training state
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.early_stopping_counter = 0
    
    def save_checkpoint(self, epoch: int, is_best: bool = False) -> str:
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch
            is_best: Whether this is best checkpoint
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint_dir = self.experiment_dir / 'checkpoints'
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }
        
        if is_best:
            path = checkpoint_dir / 'best_model.pt'
        else:
            path = checkpoint_dir / f'checkpoint_epoch_{epoch:03d}.pt'
        
        torch.save(checkpoint, path)
        return str(path)
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Epoch number
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 0)
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        
        return epoch
    
    def should_early_stop(self, val_loss: float, patience: int = 30) -> bool:
        """
        Check if should early stop.
        
        Args:
            val_loss: Current validation loss
            patience: Patience for early stopping
            
        Returns:
            True if should stop
        """
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.early_stopping_counter = 0
            return False
        else:
            self.early_stopping_counter += 1
            return self.early_stopping_counter >= patience
    
    def log_metrics(self, epoch: int, train_loss: float, val_loss: float) -> None:
        """
        Log training metrics.
        
        Args:
            epoch: Current epoch
            train_loss: Training loss
            val_loss: Validation loss
        """
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        
        log_path = self.experiment_dir / 'training_log.jsonl'
        with open(log_path, 'a') as f:
            log_entry = {
                'epoch': epoch,
                'train_loss': float(train_loss),
                'val_loss': float(val_loss)
            }
            f.write(json.dumps(log_entry) + '\n')
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get device information."""
        return {
            'device': str(self.device),
            'cuda_available': torch.cuda.is_available(),
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
        }


class DetectionTrainer(BaseTrainer):
    """Trainer for aneurysm detection (Stage 1)."""
    
    def __init__(self,
                 model: nn.Module,
                 device: torch.device,
                 experiment_dir: str,
                 optimizer: torch.optim.Optimizer,
                 loss_fn: nn.Module,
                 scheduler: Optional[Any] = None,
                 seed: int = 42):
        """
        Initialize detection trainer.
        
        Args:
            model: PointNet++ model
            device: torch.device
            experiment_dir: Experiment directory
            optimizer: Optimizer
            loss_fn: Loss function
            scheduler: Learning rate scheduler
            seed: Random seed
        """
        super().__init__(model, device, experiment_dir, seed)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for batch in pbar:
            # Move to device
            points = batch['points'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(points)
            loss = self.loss_fn(logits, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Validate model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            (avg_loss, predictions, targets)
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        all_logits = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                points = batch['points'].to(self.device)
                labels = batch['label'].to(self.device)
                
                logits = self.model(points)
                loss = self.loss_fn(logits, labels)
                
                total_loss += loss.item()
                num_batches += 1
                
                # Store predictions
                probs = torch.softmax(logits, dim=1)[:, 1]
                all_logits.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        avg_loss = total_loss / num_batches
        predictions = np.concatenate(all_logits)
        targets = np.concatenate(all_labels)
        
        return avg_loss, predictions, targets
    
    def fit(self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            num_epochs: int = 200,
            early_stopping_patience: int = 30) -> Dict[str, Any]:
        """
        Train model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs
            early_stopping_patience: Early stopping patience
            
        Returns:
            Dictionary with training results
        """
        print(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_preds, val_targets = self.validate(val_loader)
            
            # Log
            self.log_metrics(epoch, train_loss, val_loss)
            
            print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            # Learning rate schedule
            if self.scheduler:
                self.scheduler.step()
            
            # Checkpointing
            is_best = val_loss < self.best_val_loss
            if is_best or (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, is_best=is_best)
            
            # Early stopping
            if self.should_early_stop(val_loss, early_stopping_patience):
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        best_checkpoint_path = self.experiment_dir / 'checkpoints' / 'best_model.pt'
        if best_checkpoint_path.exists():
            self.load_checkpoint(str(best_checkpoint_path))
        
        return {
            'num_epochs': epoch + 1,
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }


class PINNTrainer(BaseTrainer):
    """Trainer for Physics-Informed Neural Network (Stage 2)."""
    
    def __init__(self,
                 model: nn.Module,
                 pinn_calculator,
                 device: torch.device,
                 experiment_dir: str,
                 optimizer: torch.optim.Optimizer,
                 loss_fn: nn.Module,
                 seed: int = 42):
        """
        Initialize PINN trainer.
        
        Args:
            model: PINN model
            pinn_calculator: NavierStokesResidualCalculator
            device: torch.device
            experiment_dir: Experiment directory
            optimizer: Optimizer
            loss_fn: Loss function
            seed: Random seed
        """
        super().__init__(model, device, experiment_dir, seed)
        self.pinn_calculator = pinn_calculator
        self.optimizer = optimizer
        self.loss_fn = loss_fn
    
    def train_epoch(self, x_collocation: torch.Tensor,
                   x_boundary: torch.Tensor,
                   loss_weights: Dict[str, float]) -> float:
        """
        Train for one epoch with collocation points.
        
        Args:
            x_collocation: Collocation points (B, 4)
            x_boundary: Boundary points (B, 4)
            loss_weights: Loss component weights
            
        Returns:
            Training loss
        """
        self.model.train()
        x_collocation.requires_grad = True
        
        # Forward pass
        self.optimizer.zero_grad()
        
        # PINN predictions at collocation points
        u_pred = self.model(x_collocation)
        
        # Compute physics residuals
        residuals = self.pinn_calculator.compute_residuals(
            u_pred, u_pred[:, 3:4], x_collocation
        )
        
        # Physics loss
        physics_loss = (
            loss_weights.get('continuity', 1.0) * torch.mean(residuals['continuity'] ** 2) +
            loss_weights.get('momentum', 1.0) * (
                torch.mean(residuals['momentum_x'] ** 2) +
                torch.mean(residuals['momentum_y'] ** 2) +
                torch.mean(residuals['momentum_z'] ** 2)
            ) / 3.0
        )
        
        # Boundary condition loss (simplified)
        u_bc = self.model(x_boundary)
        bc_loss = torch.mean(u_bc[:, :3] ** 2)  # Wall no-slip condition
        
        # Total loss
        total_loss = physics_loss + loss_weights.get('bc', 1.0) * bc_loss
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item()
    
    def fit(self,
            x_collocation: torch.Tensor,
            x_boundary: torch.Tensor,
            num_epochs: int = 14100,
            loss_weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Train PINN.
        
        Args:
            x_collocation: Collocation points
            x_boundary: Boundary points
            num_epochs: Number of epochs
            loss_weights: Loss weights
            
        Returns:
            Training results
        """
        if loss_weights is None:
            loss_weights = {
                'continuity': 1.0,
                'momentum': 1.0,
                'bc': 1.0
            }
        
        print(f"Training PINN for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            loss = self.train_epoch(x_collocation, x_boundary, loss_weights)
            self.train_losses.append(loss)
            
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{num_epochs} | Loss: {loss:.2e}")
            
            if (epoch + 1) % 1000 == 0:
                self.save_checkpoint(epoch)
        
        return {
            'num_epochs': num_epochs,
            'final_loss': self.train_losses[-1],
            'train_losses': self.train_losses
        }


def create_optimizer(model: nn.Module, 
                    optimizer_name: str = 'adam',
                    learning_rate: float = 2e-5,
                    weight_decay: float = 1e-4) -> torch.optim.Optimizer:
    """
    Create optimizer.
    
    Args:
        model: Model to optimize
        optimizer_name: 'adam' or 'sgd'
        learning_rate: Learning rate
        weight_decay: Weight decay
        
    Returns:
        Optimizer instance
    """
    if optimizer_name.lower() == 'adam':
        return torch.optim.Adam(model.parameters(),
                               lr=learning_rate,
                               weight_decay=weight_decay)
    elif optimizer_name.lower() == 'sgd':
        return torch.optim.SGD(model.parameters(),
                              lr=learning_rate,
                              weight_decay=weight_decay,
                              momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def create_scheduler(optimizer: torch.optim.Optimizer,
                    scheduler_name: str = 'cosine',
                    num_epochs: int = 200) -> Optional[Any]:
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: Optimizer
        scheduler_name: Scheduler type
        num_epochs: Number of epochs
        
    Returns:
        Scheduler instance or None
    """
    if scheduler_name.lower() == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=num_epochs, T_mult=1, eta_min=1e-7
        )
    elif scheduler_name.lower() == 'exponential':
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    elif scheduler_name.lower() == 'step':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    else:
        return None
