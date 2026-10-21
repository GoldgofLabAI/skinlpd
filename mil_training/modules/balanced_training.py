"""
Balanced training utilities for handling class imbalance in classification tasks.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import WeightedRandomSampler
from collections import Counter


def calculate_class_weights(labels, method='inverse_frequency', smooth=0.0):
    """
    Calculate class weights for balanced training.
    
    Args:
        labels (list or array): Class labels for the dataset
        method (str): Method for calculating weights
            - 'inverse_frequency': 1 / class_frequency
            - 'effective_samples': Effective number of samples approach
            - 'balanced': sklearn-style balanced weights
        smooth (float): Smoothing factor to prevent extreme weights
    
    Returns:
        torch.Tensor: Class weights tensor
    """
    # Count classes
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    unique_classes = np.unique(labels)
    class_counts = np.array([np.sum(labels == cls) for cls in unique_classes])
    n_samples = len(labels)
    n_classes = len(unique_classes)
    
    if method == 'inverse_frequency':
        # Simple inverse frequency
        weights = 1.0 / (class_counts + smooth)
    elif method == 'balanced':
        # sklearn-style balanced weights: n_samples / (n_classes * class_count)
        weights = n_samples / (n_classes * class_counts)
    elif method == 'effective_samples':
        # Effective number of samples (good for extreme imbalance)
        beta = 0.999  # Hyperparameter, can be tuned
        effective_num = 1.0 - np.power(beta, class_counts)
        weights = (1.0 - beta) / effective_num
    else:
        raise ValueError(f"Unknown weighting method: {method}")
    
    # Normalize weights so they sum to number of classes
    weights = weights / weights.sum() * n_classes
    
    return torch.FloatTensor(weights)


def create_weighted_sampler(labels, replacement=True):
    """
    Create a WeightedRandomSampler for balanced sampling.
    
    Args:
        labels (list or array): Class labels for the dataset
        replacement (bool): Whether to sample with replacement
    
    Returns:
        WeightedRandomSampler: Configured sampler for balanced sampling
    """
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    # Calculate weights for each sample
    class_counts = Counter(labels)
    class_weights = {cls: 1.0/count for cls, count in class_counts.items()}
    sample_weights = [class_weights[label] for label in labels]
    
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels),
        replacement=replacement
    )


def get_balanced_loss_function(class_counts=None, class_weights=None, 
                             weight_method='balanced', pos_weight=None,
                             loss_type='crossentropy', device=None):
    """
    Get a loss function configured for class imbalance.
    
    NOTE: This function now primarily supports CrossEntropyLoss since our models
    output [batch_size, num_classes] even for binary classification.
    
    Args:
        class_counts (dict or list): Dictionary {class: count} or list of counts
        class_weights (torch.Tensor): Pre-computed class weights
        weight_method (str): Method for calculating weights if class_weights not provided
        pos_weight (float): For binary classification (converted to class weights)
        loss_type (str): Type of loss function ('crossentropy', 'bce') - 'bce' not recommended
        device (torch.device): Device to move weights to
    
    Returns:
        nn.Module: Configured loss function
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Always use CrossEntropyLoss for compatibility with our model architecture
    if class_weights is not None:
        weights = class_weights.to(device)
    elif class_counts is not None:
        if isinstance(class_counts, dict):
            labels = []
            for cls, count in class_counts.items():
                labels.extend([cls] * count)
        else:
            labels = []
            for cls_idx, count in enumerate(class_counts):
                labels.extend([cls_idx] * count)
        weights = calculate_class_weights(labels, method=weight_method).to(device)
    elif pos_weight is not None and loss_type.lower() == 'bce':
        # Convert pos_weight to class weights for CrossEntropyLoss
        # pos_weight = neg_count / pos_count, so class_weight[1] = pos_weight, class_weight[0] = 1.0
        weights = torch.tensor([1.0, pos_weight]).to(device)
    else:
        weights = None
    
    return nn.CrossEntropyLoss(weight=weights)


class BalancedTrainingConfig:
    """Configuration class for balanced training options."""
    
    def __init__(self, 
                 enable_balancing=False,
                 balancing_method='class_weighted_loss',
                 sampling_strategy='natural',
                 weight_method='balanced',
                 pos_weight=None,
                 smooth_weights=0.0):
        """
        Initialize balanced training configuration.
        
        Args:
            enable_balancing (bool): Whether to enable class balancing
            balancing_method (str): Method for balancing
                - 'class_weighted_loss': Use class weights in loss function
                - 'weighted_sampling': Use weighted random sampling
                - 'both': Use both weighted loss and sampling
            sampling_strategy (str): Sampling strategy
                - 'natural': Natural distribution sampling
                - 'balanced': Balanced sampling (requires weighted_sampling)
            weight_method (str): Method for calculating class weights
            pos_weight (float): Specific positive class weight for binary classification
            smooth_weights (float): Smoothing factor for class weights
        """
        self.enable_balancing = enable_balancing
        self.balancing_method = balancing_method
        self.sampling_strategy = sampling_strategy
        self.weight_method = weight_method
        self.pos_weight = pos_weight
        self.smooth_weights = smooth_weights
    
    def get_loss_function(self, class_counts, num_classes, device=None):
        """Get appropriate loss function based on configuration."""
        # IMPORTANT: Always use CrossEntropyLoss since our model outputs [batch_size, num_classes]
        # even for binary classification (num_classes=2)
        
        if not self.enable_balancing or self.balancing_method == 'weighted_sampling':
            # No class weighting in loss function
            return nn.CrossEntropyLoss()
        
        # Use class-weighted CrossEntropyLoss for all cases
        return get_balanced_loss_function(
            class_counts=class_counts,
            weight_method=self.weight_method,
            loss_type='crossentropy',
            device=device
        )
    
    def get_sampler(self, labels):
        """Get appropriate sampler based on configuration."""
        if (not self.enable_balancing or 
            self.balancing_method == 'class_weighted_loss' or
            self.sampling_strategy == 'natural'):
            return None  # Use default random sampling
        
        # Use weighted sampling for balancing
        return create_weighted_sampler(labels, replacement=True)
    
    def print_config(self):
        """Print the current configuration."""
        print(f"Balanced Training Configuration:")
        print(f"  Balancing enabled: {self.enable_balancing}")
        if self.enable_balancing:
            print(f"  Balancing method: {self.balancing_method}")
            print(f"  Sampling strategy: {self.sampling_strategy}")
            print(f"  Weight method: {self.weight_method}")
            if self.pos_weight is not None:
                print(f"  Positive class weight: {self.pos_weight}")
            print(f"  Weight smoothing: {self.smooth_weights}")


def get_metric_recommendations_for_imbalance(imbalance_ratio, num_classes):
    """
    Get metric recommendations based on imbalance ratio and number of classes.
    
    Args:
        imbalance_ratio (float): Ratio of majority to minority class
        num_classes (int): Number of classes
    
    Returns:
        dict: Recommendations for metrics and training strategies
    """
    recommendations = {
        'primary_metric': 'val_f1_macro',
        'secondary_metrics': ['val_auprc_macro', 'val_mcc'],
        'balancing_method': 'class_weighted_loss',
        'weight_method': 'balanced'
    }
    
    if num_classes == 2:
        # Binary classification
        if imbalance_ratio < 3:
            recommendations.update({
                'primary_metric': 'val_f1_macro',
                'secondary_metrics': ['val_auprc_macro', 'val_roc_auc_macro'],
                'balancing_method': None,  # No balancing needed
                'message': '🟢 Relatively balanced - standard training should work well'
            })
        elif imbalance_ratio < 10:
            recommendations.update({
                'primary_metric': 'val_auprc_macro',  # AUPRC better for moderate imbalance
                'secondary_metrics': ['val_f1_macro', 'val_mcc', 'val_sensitivity'],
                'balancing_method': 'class_weighted_loss',
                'message': '🟡 Moderate imbalance - use AUPRC and class weighting'
            })
        else:
            recommendations.update({
                'primary_metric': 'val_auprc_macro',  # AUPRC essential for high imbalance
                'secondary_metrics': ['val_sensitivity', 'val_specificity', 'val_mcc'],
                'balancing_method': 'class_weighted_loss',
                'weight_method': 'effective_samples',
                'message': '🔴 High imbalance - AUPRC critical, consider effective samples weighting'
            })
    else:
        # Multiclass classification
        if imbalance_ratio < 3:
            recommendations.update({
                'primary_metric': 'val_f1_macro',
                'secondary_metrics': ['val_f1_micro', 'val_acc'],
                'balancing_method': None,
                'message': '🟢 Relatively balanced multiclass - macro F1 should work well'
            })
        elif imbalance_ratio < 10:
            recommendations.update({
                'primary_metric': 'val_f1_macro',  # Macro F1 gives equal weight to all classes
                'secondary_metrics': ['val_auprc_macro', 'val_mcc'],
                'balancing_method': 'class_weighted_loss',
                'message': '🟡 Imbalanced multiclass - use macro F1 and class weighting'
            })
        else:
            recommendations.update({
                'primary_metric': 'val_f1_macro',
                'secondary_metrics': ['val_auprc_macro', 'val_mcc'],
                'balancing_method': 'class_weighted_loss',
                'weight_method': 'effective_samples',
                'message': '🔴 Highly imbalanced multiclass - macro F1 + effective samples weighting'
            })
    
    return recommendations


def analyze_class_distribution(labels, class_names=None):
    """
    Analyze and print class distribution statistics.
    
    Args:
        labels (list or array): Class labels
        class_names (list): Optional class names for better reporting
    
    Returns:
        dict: Class distribution statistics with recommendations
    """
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    unique_classes, counts = np.unique(labels, return_counts=True)
    total_samples = len(labels)
    
    print(f"\nClass Distribution Analysis:")
    print(f"Total samples: {total_samples}")
    print(f"Number of classes: {len(unique_classes)}")
    
    distribution = {}
    for i, (cls, count) in enumerate(zip(unique_classes, counts)):
        class_name = class_names[cls] if class_names else f"Class {cls}"
        percentage = (count / total_samples) * 100
        ratio = total_samples / count
        
        distribution[cls] = {
            'count': count,
            'percentage': percentage,
            'inverse_ratio': ratio
        }
        
        print(f"  {class_name}: {count} samples ({percentage:.1f}%) - ratio 1:{ratio:.1f}")
    
    # Calculate imbalance metrics
    max_count = max(counts)
    min_count = min(counts)
    imbalance_ratio = max_count / min_count
    num_classes = len(unique_classes)
    
    print(f"\nImbalance Metrics:")
    print(f"  Imbalance ratio (max/min): {imbalance_ratio:.2f}")
    print(f"  Number of classes: {num_classes}")
    
    # Get recommendations based on imbalance and class count
    recommendations = get_metric_recommendations_for_imbalance(imbalance_ratio, num_classes)
    
    print(f"\n📊 RECOMMENDATIONS:")
    print(f"  {recommendations['message']}")
    print(f"  Recommended primary metric: {recommendations['primary_metric']}")
    print(f"  Secondary metrics: {', '.join(recommendations['secondary_metrics'])}")
    
    if recommendations['balancing_method']:
        print(f"  Recommended balancing: {recommendations['balancing_method']}")
        if 'weight_method' in recommendations:
            print(f"  Weight method: {recommendations['weight_method']}")
    else:
        print(f"  Balancing: Not needed")
    
    distribution.update({
        'imbalance_ratio': imbalance_ratio,
        'num_classes': num_classes,
        'recommendations': recommendations
    })
    
    return distribution
