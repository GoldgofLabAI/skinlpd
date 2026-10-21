"""
Threshold optimization utilities for finding optimal operating points.

This module provides functions to find optimal classification thresholds
using various methods (Youden's J, F1-maximization, fixed sensitivity, etc.).
"""

import numpy as np
import torch
from sklearn.metrics import roc_curve, precision_recall_curve


def find_optimal_threshold(model, val_loader, class_names, device, method='youden'):
    """
    Find optimal threshold on validation set.
    
    Args:
        model: Trained model
        val_loader: Validation dataloader
        class_names: List of class names
        device: Device to use for evaluation
        method: 'youden', 'f1', or 'sensitivity_95'
    
    Returns:
        float: Optimal threshold
    """
    model.eval()
    
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for features, labels in val_loader:
            features = features.to(device)
            labels = labels.to(device).long()
            
            outputs = model(features)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # For binary classification, use positive class probabilities
    if len(class_names) == 2:
        y_proba = all_probs[:, 1]  # Positive class probabilities
    else:
        raise ValueError("Threshold optimization currently only supports binary classification")
    
    if method == 'youden':
        # Youden's J: maximize (sensitivity + specificity - 1)
        fpr, tpr, thresholds = roc_curve(all_labels, y_proba)
        youden_j = tpr - fpr
        optimal_idx = np.argmax(youden_j)
        optimal_threshold = thresholds[optimal_idx]
        
    elif method == 'f1':
        # Maximize F1 score
        precision, recall, thresholds = precision_recall_curve(all_labels, y_proba)
        # Convert to F1 scores
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
        
    elif method == 'sensitivity_95':
        # Find threshold that gives ~95% sensitivity
        fpr, tpr, thresholds = roc_curve(all_labels, y_proba)
        # Find threshold closest to 0.95 sensitivity
        target_sensitivity = 0.95
        optimal_idx = np.argmin(np.abs(tpr - target_sensitivity))
        optimal_threshold = thresholds[optimal_idx]
    else:
        raise ValueError(f"Unknown method: {method}. Choose from: 'youden', 'f1', 'sensitivity_95'")
    
    return optimal_threshold

