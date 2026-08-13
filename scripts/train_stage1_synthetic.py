"""
Stage 1: Train PointNet++ for aneurysm detection on synthetic point clouds.

This script:
1. Generates synthetic point cloud dataset with binary labels (normal vessel / aneurysm)
2. Splits data into train/val/test with patient-level grouping (no leakage)
3. Trains PointNet++ classification model
4. Evaluates metrics: ROC-AUC, PR-AUC, accuracy, sensitivity, specificity, F1
5. Saves checkpoint and metrics to JSON after each epoch

Output:
  experiments/T1_detection_baseline/
  ├── checkpoint_latest.pt       (after each epoch)
  ├── checkpoint_best.pt         (best validation accuracy)
  ├── training_history.json      (updated each epoch)
  └── metrics.json               (updated on test set)

CPU-OPTIMIZED: Fast smoke test with defaults:
  python scripts/train_stage1_synthetic.py --smoke
  
Full training (CPU):
  python scripts/train_stage1_synthetic.py --n-samples 200 --batch-size 4 --epochs 10

GPU:
  python scripts/train_stage1_synthetic.py --n-samples 500 --batch-size 16 --epochs 30 --device cuda
"""

import sys
import os
import json
import argparse
import signal
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from models import PointNet2Classification
from utils import set_random_seed, get_device, check_split_leakage
from evaluation import ClassificationMetrics, compute_all_metrics


class SyntheticPointCloudDataset(Dataset):
    """Synthetic point cloud dataset with binary labels."""
    
    def __init__(self, samples, augmentation=False):
        self.samples = samples
        self.augmentation = augmentation
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        points = sample['points']
        # Handle NumPy 2.x compatibility: use torch.tensor instead of torch.from_numpy
        if isinstance(points, np.ndarray):
            points = torch.tensor(points, dtype=torch.float32)
        else:
            points = torch.as_tensor(points, dtype=torch.float32)
        
        label = sample['label']
        patient_id = sample['patient_id']
        
        if self.augmentation:
            points = self._augment_tensor(points)
        
        return {
            'points': points,
            'label': torch.tensor(label, dtype=torch.long),
            'patient_id': patient_id
        }
    
    def _augment_tensor(self, points):
        """Apply simple data augmentation to tensor."""
        # Random rotation around z-axis
        angle = torch.rand(1).item() * np.pi * 2
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rot = torch.tensor([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]], 
                          dtype=torch.float32)
        points[:, :3] = points[:, :3] @ rot.T
        
        # Jitter
        points[:, :3] += torch.randn_like(points[:, :3]) * 0.01
        
        return points


def generate_synthetic_dataset(n_samples=40, n_pos=None, n_points=2048):
    """Generate synthetic point cloud dataset with binary labels.
    
    Args:
        n_samples: Total number of samples
        n_pos: Number of positive (aneurysm) samples; defaults to n_samples//2
        n_points: Points per cloud (default 2048 for CPU smoke test; use 8192 for full training)
    """
    if n_pos is None:
        n_pos = n_samples // 2
    
    samples = []
    
    for i in range(n_samples):
        patient_id = f"synth_{i // 10}"  # Group into patients
        
        if i < n_pos:
            # Aneurysm: elongated vessel with spherical bulge
            base = np.random.randn(n_points // 2, 3) * 0.03
            base[:, 2] = np.linspace(-0.1, 0.1, n_points // 2)
            
            bulge = np.random.randn(n_points // 2, 3) * 0.04 + np.array([0.08, 0, 0])
            points = np.vstack([base, bulge]).astype(np.float32)
            label = 1
        else:
            # Normal: smooth elongated vessel
            t = np.linspace(0, 4 * np.pi, n_points)
            points = np.column_stack([
                0.05 * np.cos(t),
                0.05 * np.sin(t),
                np.linspace(-0.1, 0.1, n_points)
            ]).astype(np.float32)
            label = 0
        
        # Add normals (simple random normals for synthetic data)
        normals = np.random.randn(n_points, 3)
        normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)
        
        points_with_normals = np.hstack([points, normals]).astype(np.float32)
        
        samples.append({
            'points': points_with_normals,
            'label': label,
            'patient_id': patient_id
        })
    
    return samples


def split_data(samples, train_ratio=0.7, val_ratio=0.15):
    """Split dataset by patient to avoid leakage.
    
    For very small datasets (smoke test), test may be empty; evaluation falls back to val.
    """
    patient_ids = [s['patient_id'] for s in samples]
    unique_patients = sorted(set(patient_ids))
    
    n = len(unique_patients)
    
    # For smoke/small datasets: prioritize train + val, allow empty test
    if n <= 3:
        # Minimal split: 1-2 train, 1 val, rest is test (may be 0)
        n_train = max(1, (n + 1) // 2)
        n_val = max(1, n - n_train)
        n_test = 0
    else:
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))
        n_test = max(0, n - n_train - n_val)
    
    np.random.seed(42)
    perm = np.random.permutation(n)
    
    train_patients = set(unique_patients[i] for i in perm[:n_train])
    val_patients = set(unique_patients[i] for i in perm[n_train:n_train+n_val])
    test_patients = set(unique_patients[i] for i in perm[n_train+n_val:]) if n_test > 0 else set()
    
    # Verify no leakage
    leakage = check_split_leakage(
        list(train_patients), list(val_patients), list(test_patients)
    )
    print(leakage['report'])
    
    train_data = [s for s in samples if s['patient_id'] in train_patients]
    val_data = [s for s in samples if s['patient_id'] in val_patients]
    test_data = [s for s in samples if s['patient_id'] in test_patients]
    
    return train_data, val_data, test_data


def train_stage1(
    n_samples=40,
    batch_size=4,
    epochs=3,
    learning_rate=1e-3,
    n_points=2048,
    device_name='cpu',
    experiment_dir='experiments/T1_detection_baseline'
):
    """Train PointNet++ for Stage 1 detection.
    
    Args:
        n_samples: Number of synthetic samples (default 40 for CPU smoke)
        batch_size: Batch size (default 4 for CPU)
        epochs: Number of epochs (default 3 for smoke test)
        learning_rate: Learning rate
        n_points: Points per cloud (default 2048 for CPU, use 8192 for full)
        device_name: 'cpu', 'cuda', or 'auto' (auto=use CUDA if available)
        experiment_dir: Output directory
    """
    
    print("\n" + "="*80)
    print("  STAGE 1: ANEURYSM DETECTION (SYNTHETIC DATA)")
    print("="*80)
    print(f"  Config: n_samples={n_samples}, batch_size={batch_size}, epochs={epochs}")
    print(f"  Points per cloud: {n_points}")
    
    # Setup
    set_random_seed(42)
    if device_name == 'auto':
        device = get_device('cuda')
    else:
        device = torch.device(device_name)
    print(f"  Device: {device}")
    experiment_dir = Path(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate and split data
    print("\n[1/5] Generating synthetic dataset...")
    samples = generate_synthetic_dataset(n_samples=n_samples, n_points=n_points)
    print(f"  Generated {len(samples)} samples ({n_points} points each)")
    
    print("\n[2/5] Splitting data (patient-level, no leakage)...")
    train_data, val_data, test_data = split_data(samples)
    print(f"  Train: {len(train_data)} samples")
    print(f"  Val:   {len(val_data)} samples")
    print(f"  Test:  {len(test_data)} samples")
    
    # Create dataloaders
    train_dataset = SyntheticPointCloudDataset(train_data, augmentation=True)
    val_dataset = SyntheticPointCloudDataset(val_data, augmentation=False)
    test_dataset = SyntheticPointCloudDataset(test_data, augmentation=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Model
    print("\n[3/5] Initializing PointNet2Classification...")
    model = PointNet2Classification(in_channels=6, num_classes=2).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model has {n_params:,} parameters")
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    print("\n[4/5] Training...")
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0
    interrupted = False
    
    def signal_handler(sig, frame):
        nonlocal interrupted
        print("\n\n⚠️  Training interrupted. Saving checkpoint...")
        interrupted = True
    
    signal.signal(signal.SIGINT, signal_handler)
    
    start_time = time.time()
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        train_samples = 0
        
        for batch_idx, batch in enumerate(train_loader):
            points = batch['points'].to(device)  # (B, N, 6)
            labels = batch['label'].to(device)   # (B,)
            
            optimizer.zero_grad()
            logits = model(points)  # (B, 2)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * len(labels)
            train_samples += len(labels)
            
            # Print batch progress for smoke tests (every batch if small, every N batches otherwise)
            n_batches = len(train_loader)
            if n_batches <= 5 or batch_idx % max(1, n_batches // 5) == 0:
                print(f"    Epoch {epoch+1}/{epochs} | Batch {batch_idx+1}/{n_batches} | "
                      f"loss={loss.item():.4f}")
        
        train_loss /= max(train_samples, 1)
        history['train_loss'].append(train_loss)
        
        # Validate
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        val_samples = 0
        
        if len(val_dataset) > 0:
            with torch.no_grad():
                for batch in val_loader:
                    points = batch['points'].to(device)
                    labels = batch['label'].to(device)
                    
                    logits = model(points)
                    loss = criterion(logits, labels)
                    val_loss += loss.item() * len(labels)
                    
                    preds = logits.argmax(dim=1)
                    val_preds.append(preds.cpu().numpy())
                    val_labels.append(labels.cpu().numpy())
                    val_samples += len(labels)
            
            val_loss /= max(val_samples, 1)
            val_preds_all = np.concatenate(val_preds)
            val_labels_all = np.concatenate(val_labels)
            val_acc = (val_preds_all == val_labels_all).mean()
        else:
            val_loss = 0.0
            val_acc = 0.0
        
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Progress print (every epoch, not every 10)
        elapsed = time.time() - start_time
        print(f"  Epoch {epoch+1:3d}/{epochs} | "
              f"train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} | "
              f"val_acc={val_acc:.4f} | "
              f"time={elapsed:.1f}s")
        
        # Save checkpoint after each epoch
        torch.save(model.state_dict(), experiment_dir / 'checkpoint_latest.pt')
        
        # Save history after each epoch
        with open(experiment_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), experiment_dir / 'checkpoint_best.pt')
        
        if interrupted:
            print(f"✅ Training interrupted at epoch {epoch+1}/{epochs}. Checkpoints saved.")
            break
    
    print(f"\nTraining completed in {time.time() - start_time:.1f}s")
    
    # Test evaluation
    print("\n[5/5] Evaluating on test set...")
    model.load_state_dict(torch.load(experiment_dir / 'checkpoint_best.pt', map_location=device))
    model.eval()
    
    # If test set is empty, use validation set for final metrics
    if len(test_data) == 0:
        print("  ⚠️  Test set is empty (expected for smoke tests). Using validation metrics.")
        eval_loader = val_loader
        eval_labels_name = "VAL"
    else:
        eval_loader = test_loader
        eval_labels_name = "TEST"
    
    test_preds = []
    test_probs = []
    test_labels = []
    
    with torch.no_grad():
        for batch in eval_loader:
            points = batch['points'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(points)
            probs = torch.softmax(logits, dim=1)
            
            test_preds.append(logits.argmax(dim=1).cpu().numpy())
            test_probs.append(probs[:, 1].cpu().numpy())  # prob of class 1
            test_labels.append(labels.cpu().numpy())
    
    # Handle empty predictions
    if len(test_preds) == 0:
        print("  ⚠️  No samples available for evaluation. Skipping metrics.")
        metrics = {
            'roc_auc': 0.0,
            'pr_auc': 0.0,
            'accuracy': 0.0,
            'sensitivity': 0.0,
            'specificity': 0.0,
            'f1': 0.0,
            'note': f'Empty {eval_labels_name} set'
        }
    else:
        test_preds = np.concatenate(test_preds)
        test_probs = np.concatenate(test_probs)
        test_labels = np.concatenate(test_labels)
        
        # Compute metrics
        metrics = compute_all_metrics(test_probs, test_labels)
        metrics['eval_set'] = eval_labels_name
    
    print(f"  ROC-AUC:    {metrics.get('roc_auc', 0.0):.4f}")
    print(f"  PR-AUC:     {metrics.get('pr_auc', 0.0):.4f}")
    print(f"  Accuracy:   {metrics.get('accuracy', 0.0):.4f}")
    print(f"  Sensitivity:{metrics.get('sensitivity', 0.0):.4f}")
    print(f"  Specificity:{metrics.get('specificity', 0.0):.4f}")
    print(f"  F1:         {metrics.get('f1', 0.0):.4f}")
    
    # Save
    with open(experiment_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_to_serializable(obj):
        """Convert numpy arrays and other non-serializable types to JSON-compatible types."""
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj
    
    metrics_serializable = convert_to_serializable(metrics)
    with open(experiment_dir / 'metrics.json', 'w') as f:
        json.dump(metrics_serializable, f, indent=2)
    
    print(f"\n✅ Experiment complete. Results saved to {experiment_dir}")
    print("="*80 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Stage 1 Training: PointNet++ on Synthetic Data (CPU-Optimized)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  SMOKE TEST on CPU (16 samples, 256 pts, batch=2, 1 epoch, ~30-60s):
    python scripts/train_stage1_synthetic.py --smoke
  
  Small CPU training (80 samples, 2048 pts, 5 epochs, ~5-10 min):
    python scripts/train_stage1_synthetic.py
  
  Medium CPU training (200 samples, 2048 pts, 10 epochs):
    python scripts/train_stage1_synthetic.py --n-samples 200 --epochs 10
  
  Full GPU training (500 samples, 8192 pts, 30 epochs):
    python scripts/train_stage1_synthetic.py --n-samples 500 --n-points 8192 --batch-size 16 --epochs 30 --device cuda
        """
    )
    
    parser.add_argument('--smoke', action='store_true',
                        help='Ultra-light smoke test (16 samples, 256 pts, batch=2, epochs=1, ~30-60s on CPU)')
    parser.add_argument('--n-samples', type=int, default=None,
                        help='Number of synthetic samples (default: 24 if --smoke, else 80)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size (default: 2 if --smoke, else 4)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (default: 2 if --smoke, else 5)')
    parser.add_argument('--n-points', type=int, default=None,
                        help='Points per cloud (default: 512 if --smoke, else 2048)')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device: cpu, cuda, or auto (default: cpu)')
    parser.add_argument('--exp-dir', type=str, default='experiments/T1_detection_baseline',
                        help='Experiment directory')
    
    args = parser.parse_args()
    
    # Apply smoke test defaults if requested
    if args.smoke:
        n_samples = args.n_samples or 16        # ULTRA-LIGHT: 16 samples (2 patients)
        batch_size = args.batch_size or 2       # ULTRA-LIGHT: batch size 2 (min for BatchNorm)
        epochs = args.epochs or 1                # ULTRA-LIGHT: 1 epoch only
        n_points = args.n_points or 256         # ULTRA-LIGHT: 256 points per cloud
        experiment_dir = 'experiments/T1_smoke'  # Smoke results go to T1_smoke
        print("\n[SMOKE TEST MODE] Ultra-light CPU validation (16 samples, 256 pts, batch=2, 1 epoch)")
        print(f"  Expected time: ~30-60 seconds on CPU\n")
    else:
        n_samples = args.n_samples or 80        # Small: good for CPU testing
        batch_size = args.batch_size or 4       # Small: batch size 4
        epochs = args.epochs or 5                # Small: 5 epochs
        n_points = args.n_points or 2048        # Small: 2048 points per cloud
        experiment_dir = args.exp_dir            # Use specified or default dir
    
    train_stage1(
        n_samples=n_samples,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=args.learning_rate,
        n_points=n_points,
        device_name=args.device,
        experiment_dir=experiment_dir
    )
