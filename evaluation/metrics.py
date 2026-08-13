"""
Evaluation metrics for NeuroFlow-Diagnostics.
Includes classification metrics, calibration analysis, and clinical utility metrics.
"""

from typing import Dict, Tuple, Optional, List

import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score, auc, roc_curve, precision_recall_curve,
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, brier_score_loss, log_loss
)
from scipy import stats


class ClassificationMetrics:
    """Compute classification metrics."""
    
    @staticmethod
    def compute_metrics(predictions: np.ndarray, targets: np.ndarray,
                       threshold: float = 0.5) -> Dict[str, float]:
        """
        Compute comprehensive classification metrics.
        
        Args:
            predictions: Predicted probabilities (N,) for binary classification
            targets: Ground truth binary labels (N,)
            threshold: Classification threshold
            
        Returns:
            Dictionary of metrics
        """
        # Ensure binary predictions
        binary_preds = (predictions >= threshold).astype(int)
        
        # Compute metrics
        metrics = {
            'accuracy': accuracy_score(targets, binary_preds),
            'sensitivity': recall_score(targets, binary_preds, zero_division=0),
            'specificity': recall_score(1 - targets, 1 - binary_preds, zero_division=0),
            'precision': precision_score(targets, binary_preds, zero_division=0),
            'recall': recall_score(targets, binary_preds, zero_division=0),
            'f1': f1_score(targets, binary_preds, zero_division=0),
            'brier_score': brier_score_loss(targets, predictions),
            'log_loss': log_loss(targets, np.clip(predictions, 1e-15, 1 - 1e-15))
        }
        
        # AUC metrics
        try:
            metrics['auc'] = roc_auc_score(targets, predictions)
        except:
            metrics['auc'] = 0.5
        
        try:
            precision_vals, recall_vals, _ = precision_recall_curve(targets, predictions)
            metrics['pr_auc'] = auc(recall_vals, precision_vals)
        except:
            metrics['pr_auc'] = 0.5
        
        return metrics
    
    @staticmethod
    def bootstrap_ci(predictions: np.ndarray, targets: np.ndarray,
                     n_samples: int = 1000, ci: float = 0.95) -> Dict[str, Tuple[float, float]]:
        """
        Compute bootstrap confidence intervals for metrics.
        
        Args:
            predictions: Predicted probabilities
            targets: Ground truth labels
            n_samples: Number of bootstrap samples
            ci: Confidence level (0.95 for 95% CI)
            
        Returns:
            Dictionary mapping metric names to (lower, upper) CI bounds
        """
        n = len(predictions)
        metrics_samples = []
        
        for _ in range(n_samples):
            # Resample with replacement
            indices = np.random.choice(n, n, replace=True)
            preds_sample = predictions[indices]
            targets_sample = targets[indices]
            
            try:
                auc_score = roc_auc_score(targets_sample, preds_sample)
            except:
                auc_score = 0.5
            
            metrics_samples.append({'auc': auc_score})
        
        # Compute confidence intervals
        alpha = (1 - ci) / 2
        ci_dict = {}
        
        for metric_name in metrics_samples[0].keys():
            values = [m[metric_name] for m in metrics_samples]
            lower = np.percentile(values, alpha * 100)
            upper = np.percentile(values, (1 - alpha) * 100)
            ci_dict[metric_name] = (lower, upper)
        
        return ci_dict
    
    @staticmethod
    def confusion_matrix_metrics(predictions: np.ndarray, targets: np.ndarray,
                                threshold: float = 0.5) -> Dict:
        """
        Compute confusion matrix and derived metrics.
        
        Args:
            predictions: Predicted probabilities
            targets: Ground truth labels
            threshold: Classification threshold
            
        Returns:
            Dictionary with confusion matrix and metrics
        """
        binary_preds = (predictions >= threshold).astype(int)
        cm = confusion_matrix(targets, binary_preds)
        
        tn, fp, fn, tp = cm.ravel()
        
        return {
            'confusion_matrix': cm,
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp),
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'ppv': tp / (tp + fp) if (tp + fp) > 0 else 0,  # Positive Predictive Value
            'npv': tn / (tn + fn) if (tn + fn) > 0 else 0,  # Negative Predictive Value
        }


class CalibrationMetrics:
    """Compute calibration and reliability metrics."""
    
    @staticmethod
    def expected_calibration_error(predictions: np.ndarray, targets: np.ndarray,
                                  n_bins: int = 10) -> float:
        """
        Compute Expected Calibration Error (ECE).
        
        Args:
            predictions: Predicted probabilities
            targets: Ground truth labels
            n_bins: Number of bins for calibration
            
        Returns:
            ECE value (lower is better, 0 is perfect calibration)
        """
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        ece = 0.0
        
        for i in range(n_bins):
            mask = (predictions >= bin_edges[i]) & (predictions < bin_edges[i + 1])
            if mask.sum() == 0:
                continue
            
            bin_acc = targets[mask].mean()
            bin_conf = predictions[mask].mean()
            bin_size = mask.sum()
            
            ece += (bin_size / len(predictions)) * np.abs(bin_acc - bin_conf)
        
        return ece
    
    @staticmethod
    def calibration_curve(predictions: np.ndarray, targets: np.ndarray,
                         n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute calibration curve (fraction of positives vs mean predicted probability).
        
        Args:
            predictions: Predicted probabilities
            targets: Ground truth labels
            n_bins: Number of bins
            
        Returns:
            (mean_predictions_per_bin, fraction_positives_per_bin)
        """
        bin_edges = np.linspace(0, 1, n_bins + 1)
        
        mean_preds = []
        frac_positives = []
        
        for i in range(n_bins):
            mask = (predictions >= bin_edges[i]) & (predictions < bin_edges[i + 1])
            if mask.sum() == 0:
                continue
            
            mean_preds.append(predictions[mask].mean())
            frac_positives.append(targets[mask].mean())
        
        return np.array(mean_preds), np.array(frac_positives)
    
    @staticmethod
    def platt_scaling(predictions: np.ndarray, targets: np.ndarray
                     ) -> Tuple[float, float]:
        """
        Compute Platt scaling parameters for calibration.
        
        Args:
            predictions: Predicted probabilities
            targets: Ground truth labels
            
        Returns:
            (slope, intercept) for calibration: p_calibrated = 1 / (1 + exp(-(slope*logit(p) + intercept)))
        """
        # Fit logistic regression
        from sklearn.linear_model import LogisticRegression
        
        # Convert to logit space
        clipped_preds = np.clip(predictions, 1e-15, 1 - 1e-15)
        logit_preds = np.log(clipped_preds / (1 - clipped_preds)).reshape(-1, 1)
        
        # Fit
        lr = LogisticRegression(fit_intercept=True, solver='lbfgs')
        lr.fit(logit_preds, targets)
        
        slope = lr.coef_[0, 0]
        intercept = lr.intercept_[0]
        
        return slope, intercept


class HemodynamicMetrics:
    """Metrics for validating hemodynamic simulations (PINN)."""
    
    @staticmethod
    def velocity_field_rmse(predicted: torch.Tensor, ground_truth: torch.Tensor
                           ) -> float:
        """
        Compute RMSE of velocity fields.
        
        Args:
            predicted: Predicted velocity field (N, 3)
            ground_truth: Ground truth velocity field (N, 3)
            
        Returns:
            Normalized RMSE
        """
        diff = predicted - ground_truth
        rmse = torch.sqrt(torch.mean(diff ** 2)).item()
        
        # Normalize by ground truth magnitude
        gt_mag = torch.norm(ground_truth, dim=1).mean().item()
        if gt_mag > 1e-8:
            normalized_rmse = rmse / gt_mag
        else:
            normalized_rmse = rmse
        
        return normalized_rmse
    
    @staticmethod
    def pressure_field_correlation(predicted: torch.Tensor,
                                  ground_truth: torch.Tensor) -> float:
        """
        Compute correlation coefficient of pressure fields.
        
        Args:
            predicted: Predicted pressure field (N,)
            ground_truth: Ground truth pressure field (N,)
            
        Returns:
            Correlation coefficient (R²)
        """
        pred_np = predicted.detach().cpu().numpy().flatten()
        gt_np = ground_truth.detach().cpu().numpy().flatten()
        
        correlation = np.corrcoef(pred_np, gt_np)[0, 1]
        r_squared = correlation ** 2
        
        return r_squared
    
    @staticmethod
    def mass_conservation_error(inlet_flow: float, outlet_flow: float) -> float:
        """
        Compute mass conservation error.
        
        Args:
            inlet_flow: Flow rate at inlet
            outlet_flow: Flow rate at outlet
            
        Returns:
            Mass conservation error (percentage)
        """
        if abs(inlet_flow) < 1e-8:
            return 0.0
        
        error = abs(outlet_flow - inlet_flow) / abs(inlet_flow) * 100
        return error


class ClinicalUtilityMetrics:
    """Metrics for clinical utility assessment."""
    
    @staticmethod
    def decision_curve_analysis(predictions: np.ndarray, targets: np.ndarray,
                               threshold_range: Tuple[float, float] = (0.01, 0.99),
                               n_thresholds: int = 100) -> Dict:
        """
        Compute Decision Curve Analysis (DCA).
        
        Args:
            predictions: Predicted probabilities
            targets: Ground truth binary labels
            threshold_range: Range of thresholds to evaluate
            n_thresholds: Number of thresholds
            
        Returns:
            Dictionary with DCA metrics
        """
        thresholds = np.linspace(threshold_range[0], threshold_range[1], n_thresholds)
        net_benefits = []
        
        total_positive = np.sum(targets)
        total_negative = len(targets) - total_positive
        
        for threshold in thresholds:
            # True positives and false positives at this threshold
            tp = np.sum((predictions >= threshold) & (targets == 1))
            fp = np.sum((predictions >= threshold) & (targets == 0))
            
            # Net benefit
            benefit = tp - (fp * threshold / (1 - threshold))
            net_benefit = benefit / len(targets)
            
            net_benefits.append(net_benefit)
        
        return {
            'thresholds': thresholds,
            'net_benefits': np.array(net_benefits),
            'optimal_threshold': thresholds[np.argmax(net_benefits)],
            'max_net_benefit': np.max(net_benefits)
        }
    
    @staticmethod
    def optimal_threshold(predictions: np.ndarray, targets: np.ndarray,
                         metric: str = 'youden') -> float:
        """
        Find optimal classification threshold.
        
        Args:
            predictions: Predicted probabilities
            targets: Ground truth labels
            metric: 'youden' (sensitivity + specificity - 1) or 'f1'
            
        Returns:
            Optimal threshold value
        """
        if metric == 'youden':
            fpr, tpr, thresholds = roc_curve(targets, predictions)
            youden = tpr - fpr
            optimal_idx = np.argmax(youden)
            return thresholds[optimal_idx]
        
        elif metric == 'f1':
            best_f1 = 0.0
            best_threshold = 0.5
            
            for threshold in np.linspace(0, 1, 101):
                binary_preds = (predictions >= threshold).astype(int)
                f1 = f1_score(targets, binary_preds, zero_division=0)
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
            
            return best_threshold
        
        return 0.5


def compute_all_metrics(predictions: np.ndarray, targets: np.ndarray,
                       patient_ids: Optional[np.ndarray] = None) -> Dict:
    """
    Compute all evaluation metrics.
    
    Args:
        predictions: Predicted probabilities (N,)
        targets: Ground truth labels (N,)
        patient_ids: Patient IDs for stratification (optional)
        
    Returns:
        Comprehensive metrics dictionary
    """
    metrics = {}
    
    # Classification metrics
    class_metrics = ClassificationMetrics.compute_metrics(predictions, targets)
    metrics.update(class_metrics)
    
    # Calibration metrics
    metrics['ece'] = CalibrationMetrics.expected_calibration_error(predictions, targets)
    slope, intercept = CalibrationMetrics.platt_scaling(predictions, targets)
    metrics['calibration_slope'] = slope
    metrics['calibration_intercept'] = intercept
    
    # Confusion matrix
    cm_metrics = ClassificationMetrics.confusion_matrix_metrics(predictions, targets)
    metrics['confusion_matrix'] = cm_metrics['confusion_matrix']
    metrics.update({k: v for k, v in cm_metrics.items() if k != 'confusion_matrix'})
    
    # Clinical utility
    dca_metrics = ClinicalUtilityMetrics.decision_curve_analysis(predictions, targets)
    metrics.update(dca_metrics)
    
    # Bootstrap CIs
    metrics['bootstrap_ci'] = ClassificationMetrics.bootstrap_ci(predictions, targets)
    
    return metrics
