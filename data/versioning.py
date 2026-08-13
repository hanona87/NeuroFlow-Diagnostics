"""
Dataset versioning and reproducibility tracking for NeuroFlow.

Provides:
- Manifest versioning and hashing
- Dataset version tracking
- Reproducibility metadata
- Experiment-to-dataset mapping
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import pandas as pd


class DatasetVersion:
    """Represents a versioned dataset snapshot."""
    
    def __init__(
        self,
        version_id: str,
        dataset_name: str,
        description: str = ""
    ):
        """
        Initialize dataset version.
        
        Args:
            version_id: Unique version identifier (e.g., "1.0", "2024-08-13-v1")
            dataset_name: Name of dataset (e.g., "intra", "synthetic")
            description: Human-readable description
        """
        self.version_id = version_id
        self.dataset_name = dataset_name
        self.description = description
        self.created_date = datetime.now().isoformat()
        self.metadata: Dict[str, Any] = {}
        self.manifest_hashes: Dict[str, str] = {}  # split_name → hash
    
    def add_manifest_hash(self, split_name: str, manifest_hash: str):
        """Record hash of a manifest for this version."""
        self.manifest_hashes[split_name] = manifest_hash
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "version_id": self.version_id,
            "dataset_name": self.dataset_name,
            "description": self.description,
            "created_date": self.created_date,
            "metadata": self.metadata,
            "manifest_hashes": self.manifest_hashes
        }
    
    def to_json(self, filepath: str):
        """Save version metadata to JSON."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"✅ Version metadata saved to {filepath}")
    
    @classmethod
    def from_json(cls, filepath: str) -> "DatasetVersion":
        """Load version metadata from JSON."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        version = cls(
            data["version_id"],
            data["dataset_name"],
            data.get("description", "")
        )
        version.metadata = data.get("metadata", {})
        version.manifest_hashes = data.get("manifest_hashes", {})
        
        return version


class ManifestHasher:
    """Computes deterministic hashes of manifests for reproducibility."""
    
    @staticmethod
    def hash_manifest_csv(csv_path: str) -> str:
        """
        Compute hash of a CSV manifest file.
        
        Uses sorted, deterministic representation to ensure
        identical manifests always produce identical hashes.
        
        Args:
            csv_path: Path to manifest CSV
            
        Returns:
            SHA256 hash hex digest
        """
        df = pd.read_csv(csv_path)
        
        # Sort by patient/study/aneurysm for determinism
        df = df.sort_values(by=['patient_id', 'study_id', 'aneurysm_id'])
        
        # Convert to JSON string for hashing
        manifest_str = df.to_json(orient='records', default_handler=str)
        
        # Compute hash
        return hashlib.sha256(manifest_str.encode()).hexdigest()
    
    @staticmethod
    def hash_manifest_json(json_path: str) -> str:
        """
        Compute hash of a JSON manifest file.
        
        Args:
            json_path: Path to manifest JSON
            
        Returns:
            SHA256 hash hex digest
        """
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Extract entries and sort
        entries = data.get('entries', [])
        entries = sorted(entries, key=lambda e: (e['patient_id'], e['study_id'], e['aneurysm_id']))
        
        # Convert to JSON string
        manifest_str = json.dumps(entries, sort_keys=True, default=str)
        
        # Compute hash
        return hashlib.sha256(manifest_str.encode()).hexdigest()
    
    @staticmethod
    def hash_manifest_directory(manifest_dir: str) -> Dict[str, str]:
        """
        Compute hashes for all manifests in a directory.
        
        Args:
            manifest_dir: Directory containing manifest files
            
        Returns:
            Dict mapping filename → hash
        """
        manifest_dir = Path(manifest_dir)
        hashes = {}
        
        for manifest_file in manifest_dir.glob("*.csv"):
            hashes[manifest_file.name] = ManifestHasher.hash_manifest_csv(str(manifest_file))
        
        for manifest_file in manifest_dir.glob("*.json"):
            hashes[manifest_file.name] = ManifestHasher.hash_manifest_json(str(manifest_file))
        
        return hashes


class ReproducibilityCard:
    """Documents all information needed to reproduce an experiment."""
    
    def __init__(self, experiment_id: str):
        """
        Initialize reproducibility card.
        
        Args:
            experiment_id: Unique experiment identifier (e.g., "T1_detector_v1")
        """
        self.experiment_id = experiment_id
        self.created_date = datetime.now().isoformat()
        self.details: Dict[str, Any] = {
            "experiment_id": experiment_id,
            "created_date": self.created_date,
            "data": {},
            "code": {},
            "training": {},
            "environment": {},
            "results": {}
        }
    
    def set_data_info(
        self,
        dataset_name: str,
        dataset_version: str,
        manifest_train_hash: str,
        manifest_val_hash: str,
        manifest_test_hash: str,
        preprocessing_version: str = "1.0",
        num_points: int = 8192,
        normalization_method: str = "unit_sphere"
    ):
        """Set data configuration information."""
        self.details["data"] = {
            "dataset_name": dataset_name,
            "dataset_version": dataset_version,
            "manifests": {
                "train_hash": manifest_train_hash,
                "val_hash": manifest_val_hash,
                "test_hash": manifest_test_hash
            },
            "preprocessing": {
                "version": preprocessing_version,
                "num_points": num_points,
                "normalization_method": normalization_method
            }
        }
    
    def set_code_info(
        self,
        git_commit: Optional[str] = None,
        git_branch: Optional[str] = None,
        python_version: str = "3.10",
        pytorch_version: str = "2.2.0",
        pytorch_geometric_version: str = "2.5.0"
    ):
        """Set code version information."""
        self.details["code"] = {
            "git_commit": git_commit,
            "git_branch": git_branch,
            "python_version": python_version,
            "pytorch_version": pytorch_version,
            "pytorch_geometric_version": pytorch_geometric_version
        }
    
    def set_training_info(
        self,
        model_name: str,
        seed: int,
        loss_function: str,
        optimizer: str,
        learning_rate: float,
        batch_size: int,
        num_epochs: int,
        early_stopping: bool = True,
        loss_weights: Optional[Dict[str, float]] = None
    ):
        """Set training configuration."""
        self.details["training"] = {
            "model_name": model_name,
            "seed": seed,
            "loss_function": loss_function,
            "optimizer": optimizer,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "early_stopping": early_stopping,
            "loss_weights": loss_weights or {}
        }
    
    def set_environment_info(
        self,
        device: str,
        cuda_version: Optional[str] = None,
        compute_capability: Optional[str] = None
    ):
        """Set environment information."""
        self.details["environment"] = {
            "device": device,
            "cuda_version": cuda_version,
            "compute_capability": compute_capability
        }
    
    def set_results_info(
        self,
        test_auc: float,
        test_pr_auc: float,
        test_accuracy: float,
        val_auc: float,
        best_epoch: int,
        total_parameters: int,
        training_time_seconds: float
    ):
        """Set results information."""
        self.details["results"] = {
            "test_auc": test_auc,
            "test_pr_auc": test_pr_auc,
            "test_accuracy": test_accuracy,
            "val_auc": val_auc,
            "best_epoch": best_epoch,
            "total_parameters": total_parameters,
            "training_time_seconds": training_time_seconds
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.details
    
    def to_json(self, filepath: str):
        """Save to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.details, f, indent=2, default=str)
        print(f"✅ Reproducibility card saved to {filepath}")
    
    @classmethod
    def from_json(cls, filepath: str) -> "ReproducibilityCard":
        """Load from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        card = cls(data["experiment_id"])
        card.details = data
        
        return card


class ExperimentRegistry:
    """Central registry of all experiments for tracking."""
    
    def __init__(self, registry_path: str = "experiments/registry.json"):
        """
        Initialize experiment registry.
        
        Args:
            registry_path: Path to save registry
        """
        self.registry_path = registry_path
        self.experiments: Dict[str, Dict[str, Any]] = {}
        
        # Load existing registry if present
        if Path(registry_path).exists():
            with open(registry_path, 'r') as f:
                self.experiments = json.load(f)
    
    def register_experiment(
        self,
        experiment_id: str,
        experiment_type: str,  # "T0", "T1", etc.
        description: str,
        status: str,  # "proposed", "in_progress", "completed", "blocked"
        blocking_reason: Optional[str] = None
    ):
        """Register an experiment."""
        self.experiments[experiment_id] = {
            "experiment_id": experiment_id,
            "experiment_type": experiment_type,
            "description": description,
            "status": status,
            "blocking_reason": blocking_reason,
            "registered_date": datetime.now().isoformat()
        }
        
        self.save()
    
    def update_experiment_status(
        self,
        experiment_id: str,
        status: str,
        blocking_reason: Optional[str] = None
    ):
        """Update experiment status."""
        if experiment_id in self.experiments:
            self.experiments[experiment_id]["status"] = status
            self.experiments[experiment_id]["blocking_reason"] = blocking_reason
            self.experiments[experiment_id]["updated_date"] = datetime.now().isoformat()
            self.save()
    
    def save(self):
        """Save registry to disk."""
        Path(self.registry_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.registry_path, 'w') as f:
            json.dump(self.experiments, f, indent=2, default=str)
    
    def get_status_summary(self) -> Dict[str, int]:
        """Get summary of experiments by status."""
        summary = {}
        
        for exp in self.experiments.values():
            status = exp["status"]
            summary[status] = summary.get(status, 0) + 1
        
        return summary
    
    def list_blocked_experiments(self) -> Dict[str, str]:
        """List all blocked experiments with reasons."""
        blocked = {}
        
        for exp_id, exp in self.experiments.items():
            if exp["status"] == "blocked":
                blocked[exp_id] = exp.get("blocking_reason", "No reason specified")
        
        return blocked
    
    def print_summary(self):
        """Print summary to console."""
        summary = self.get_status_summary()
        
        print("\n" + "="*60)
        print("EXPERIMENT REGISTRY SUMMARY")
        print("="*60)
        
        for status, count in summary.items():
            print(f"  {status:15} {count:3} experiments")
        
        blocked = self.list_blocked_experiments()
        if blocked:
            print("\n" + "="*60)
            print("BLOCKED EXPERIMENTS")
            print("="*60)
            
            for exp_id, reason in blocked.items():
                print(f"  {exp_id:30} {reason}")
        
        print()


# Module-level convenience functions

def create_reproducibility_card(
    experiment_id: str,
    config_dict: Dict[str, Any],
    git_info: Optional[Dict[str, str]] = None
) -> ReproducibilityCard:
    """
    Create a reproducibility card from a config dict and optional git info.
    
    Args:
        experiment_id: Unique experiment ID
        config_dict: Configuration dictionary
        git_info: Optional git information
        
    Returns:
        ReproducibilityCard instance
    """
    card = ReproducibilityCard(experiment_id)
    
    # Extract data config
    data_config = config_dict.get("data", {})
    card.set_data_info(
        dataset_name=data_config.get("primary_dataset", "unknown"),
        dataset_version=data_config.get("dataset_version", "1.0"),
        manifest_train_hash="",  # Will be set later
        manifest_val_hash="",
        manifest_test_hash="",
        num_points=data_config.get("num_points", 8192),
        normalization_method=data_config.get("normalization_method", "unit_sphere")
    )
    
    # Extract training config
    training_config = config_dict.get("training", {})
    card.set_training_info(
        model_name=training_config.get("model_name", "unknown"),
        seed=training_config.get("seed", 42),
        loss_function=training_config.get("loss_function", "BCE"),
        optimizer=training_config.get("optimizer", "Adam"),
        learning_rate=training_config.get("learning_rate", 0.001),
        batch_size=training_config.get("batch_size", 32),
        num_epochs=training_config.get("num_epochs", 100)
    )
    
    # Set git info if provided
    if git_info:
        card.set_code_info(
            git_commit=git_info.get("commit"),
            git_branch=git_info.get("branch"),
            python_version=git_info.get("python_version", "3.10")
        )
    
    return card
