"""
End-to-end training script for NeuroFlow-Diagnostics Stage 1 (Aneurysm Detection).
Run with: python scripts/train_stage1.py --config configs/config.yaml --experiment exp_detection_01
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import (
    set_random_seed, get_device, setup_logging, load_config,
    print_summary, create_directories
)
from models.pointnet2 import PointNet2Classification
from losses.losses import FocalLoss, WeightedCrossEntropyLoss
from evaluation.metrics import (
    ClassificationMetrics, CalibrationMetrics, ClinicalUtilityMetrics,
    compute_all_metrics
)
from trainers.trainer import (
    PointCloudDatasetLoader, DetectionTrainer,
    create_optimizer, create_scheduler
)
from data.preprocessing.preprocessing import create_synthetic_dataset


class Stage1ExperimentRunner:
    """Runner for Stage 1 (Aneurysm Detection) experiments."""
    
    def __init__(self, config: Dict[str, Any], experiment_name: str):
        """
        Initialize experiment runner.
        
        Args:
            config: Configuration dictionary
            experiment_name: Name of experiment
        """
        self.config = config
        self.experiment_name = experiment_name
        self.logger = setup_logging(level=config.get('logging', {}).get('level', 'INFO'))
        
        # Set random seed
        seed = config.get('reproducibility', {}).get('seed', 42)
        set_random_seed(seed, deterministic=config.get('reproducibility', {}).get('deterministic', True))
        
        # Setup directories
        self.experiment_dir = Path(config.get('experiments', {}).get('output_dir', './experiments')) / experiment_name
        self.results_dir = self.experiment_dir / 'results'
        self.checkpoints_dir = self.experiment_dir / 'checkpoints'
        
        directories = create_directories(str(self.experiment_dir), ['results', 'checkpoints'])
        
        # Get device
        device_str = config.get('hardware', {}).get('device', 'cuda')
        self.device = get_device(device_str)
        
        self.logger.info(f"Experiment: {experiment_name}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Experiment directory: {self.experiment_dir}")
    
    def prepare_data(self) -> tuple:
        """
        Prepare training data.
        
        Returns:
            (train_loader, val_loader, test_loader)
        """
        self.logger.info("Preparing data...")
        
        # Check if synthetic data should be used
        data_config = self.config.get('data', {})
        
        # Create synthetic dataset for demonstration
        self.logger.info("Creating synthetic dataset for demonstration...")
        dataset_path = create_synthetic_dataset(
            n_samples=100,
            n_positive=50,
            num_points=data_config.get('num_points', 8192),
            output_dir='./data/datasets'
        )
        
        # Split into train/val/test
        dataset = PointCloudDatasetLoader.load_dataset(dataset_path)
        
        n_total = len(dataset)
        n_train = int(0.7 * n_total)
        n_val = int(0.15 * n_total)
        
        from torch.utils.data import random_split
        train_dataset, val_dataset, test_dataset = random_split(
            dataset,
            [n_train, n_val, n_total - n_train - n_val]
        )
        
        # Create dataloaders
        aug_config = data_config.get('augmentation', {})
        train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=20, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=20, shuffle=False, num_workers=0)
        
        self.logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        return train_loader, val_loader, test_loader
    
    def build_model(self) -> nn.Module:
        """Build PointNet++ model."""
        self.logger.info("Building model...")
        
        stage1_config = self.config.get('stage1_detection', {})
        
        model = PointNet2Classification(
            in_channels=6,  # xyz + normals
            num_classes=2,
            use_normals=True,
            dropout_rate=stage1_config.get('architecture', {}).get('dropout_rate', 0.5)
        )
        
        # Count parameters
        n_params = sum(p.numel() for p in model.parameters())
        self.logger.info(f"Model parameters: {n_params:,}")
        
        return model
    
    def run(self) -> Dict[str, Any]:
        """Run the full experiment."""
        self.logger.info("="*80)
        self.logger.info("STAGE 1: ANEURYSM DETECTION EXPERIMENT")
        self.logger.info("="*80)
        
        # Prepare data
        train_loader, val_loader, test_loader = self.prepare_data()
        
        # Build model
        model = self.build_model()
        model = model.to(self.device)
        
        # Setup training
        stage1_config = self.config.get('stage1_detection', {})
        training_config = stage1_config.get('training', {})
        
        optimizer = create_optimizer(
            model,
            optimizer_name=training_config.get('optimizer', 'adam'),
            learning_rate=training_config.get('learning_rate', 2e-5),
            weight_decay=training_config.get('weight_decay', 1e-4)
        )
        
        scheduler = create_scheduler(
            optimizer,
            scheduler_name=training_config.get('scheduler', 'cosine'),
            num_epochs=training_config.get('epochs', 200)
        )
        
        # Loss function
        loss_type = training_config.get('loss_type', 'weighted_ce')
        if loss_type == 'focal':
            focal_config = training_config.get('focal_loss', {})
            loss_fn = FocalLoss(
                alpha=focal_config.get('alpha', 0.75),
                gamma=focal_config.get('gamma', 2.0)
            )
        else:
            loss_fn = WeightedCrossEntropyLoss(pos_weight=1.0)
        
        loss_fn = loss_fn.to(self.device)
        
        # Create trainer
        trainer = DetectionTrainer(
            model=model,
            device=self.device,
            experiment_dir=str(self.experiment_dir),
            optimizer=optimizer,
            loss_fn=loss_fn,
            scheduler=scheduler
        )
        
        # Train
        self.logger.info("Starting training...")
        training_results = trainer.fit(
            train_loader,
            val_loader,
            num_epochs=training_config.get('epochs', 200),
            early_stopping_patience=training_config.get('early_stopping_patience', 30)
        )
        
        # Evaluate on test set
        self.logger.info("Evaluating on test set...")
        test_loss, test_preds, test_targets = trainer.validate(test_loader)
        
        # Compute metrics
        test_metrics = compute_all_metrics(test_preds, test_targets)
        
        # Save results
        results = {
            'training_results': training_results,
            'test_metrics': {k: v for k, v in test_metrics.items() if k != 'confusion_matrix'},
            'confusion_matrix': test_metrics['confusion_matrix'].tolist() if isinstance(test_metrics['confusion_matrix'], np.ndarray) else test_metrics['confusion_matrix']
        }
        
        # Save to file
        results_path = self.results_dir / 'results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save predictions
        predictions_path = self.results_dir / 'predictions.npz'
        np.savez(
            predictions_path,
            predictions=test_preds,
            targets=test_targets
        )
        
        # Print summary
        print_summary("TEST SET RESULTS", {
            'AUC': f"{test_metrics.get('auc', 0):.4f}",
            'PR-AUC': f"{test_metrics.get('pr_auc', 0):.4f}",
            'Accuracy': f"{test_metrics.get('accuracy', 0):.4f}",
            'Sensitivity': f"{test_metrics.get('sensitivity', 0):.4f}",
            'Specificity': f"{test_metrics.get('specificity', 0):.4f}",
            'F1-Score': f"{test_metrics.get('f1', 0):.4f}",
            'ECE': f"{test_metrics.get('ece', 0):.4f}",
        })
        
        self.logger.info("Training complete!")
        self.logger.info(f"Results saved to {results_path}")
        
        return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Train Stage 1 Detection Model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--experiment', type=str, default='stage1_detection_01',
                       help='Experiment name')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override device if specified
    if hasattr(config, '__getitem__'):
        if 'hardware' not in config:
            config['hardware'] = {}
        config['hardware']['device'] = args.device
    
    # Run experiment
    runner = Stage1ExperimentRunner(config, args.experiment)
    results = runner.run()
    
    return results


if __name__ == '__main__':
    main()
