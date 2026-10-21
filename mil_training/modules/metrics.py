"""
Metrics calculation utilities for model evaluation.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, average_precision_score, confusion_matrix,
    matthews_corrcoef
)
from sklearn.preprocessing import label_binarize


def calculate_binary_metrics(y_true, y_pred, y_proba, class_names):
    """Calculate comprehensive metrics for binary classification."""
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    # Binary classification metrics (use 'binary' average)
    precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
    
    # ROC AUC for binary classification
    try:
        roc_auc = roc_auc_score(y_true, y_proba)
    except:
        roc_auc = float('nan')
    
    # Average Precision for binary classification
    try:
        avg_precision = average_precision_score(y_true, y_proba)
    except:
        avg_precision = float('nan')
    
    # Matthews Correlation Coefficient
    try:
        mcc = matthews_corrcoef(y_true, y_pred)
    except:
        mcc = float('nan')
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate per-class metrics for binary case
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # Create per-class metrics DataFrame
    per_class_metrics = pd.DataFrame({
        'class': class_names,
        'precision': precision_per_class,
        'recall': recall_per_class,
        'f1': f1_per_class
    })
    
    # Additional binary-specific metrics
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Same as recall
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'roc_auc': roc_auc,
        'avg_precision': avg_precision,
        'mcc': mcc,
        'confusion_matrix': cm,
        'per_class_metrics': per_class_metrics,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        'true_positives': tp
    }
    
    return metrics


def calculate_comprehensive_metrics(y_true, y_pred, y_proba, class_names):
    """Calculate comprehensive metrics for both binary and multi-class classification."""
    
    # Detect if this is binary classification
    num_classes = len(class_names)
    is_binary = num_classes == 2
    
    if is_binary:
        # For binary classification, use probability of positive class (class 1)
        if y_proba.ndim > 1:
            y_proba_binary = y_proba[:, 1]  # Probability of positive class
        else:
            y_proba_binary = y_proba
        
        # Use specialized binary metrics
        metrics = calculate_binary_metrics(y_true, y_pred, y_proba_binary, class_names)
        
        # Add some multi-class compatible metrics for consistency
        metrics['precision_macro'] = metrics['precision']
        metrics['recall_macro'] = metrics['recall']  
        metrics['f1_macro'] = metrics['f1']
        metrics['roc_auc_macro'] = metrics['roc_auc']
        metrics['avg_precision_macro'] = metrics['avg_precision']
        
        return metrics
    
    else:
        # Multi-class classification (original logic)
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        # Per-class metrics
        # Pass explicit labels so the returned arrays always have one entry per
        # class in class_names, even if some class is absent from a given
        # validation split (otherwise sklearn drops it and the per-class
        # DataFrame below fails with a length mismatch).
        class_label_indices = list(range(len(class_names)))
        precision_per_class = precision_score(y_true, y_pred, labels=class_label_indices, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, labels=class_label_indices, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, labels=class_label_indices, average=None, zero_division=0)
        
        # Macro and micro averaged metrics
        precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        precision_micro = precision_score(y_true, y_pred, average='micro', zero_division=0)
        recall_micro = recall_score(y_true, y_pred, average='micro', zero_division=0)
        f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
        
        # ROC AUC (one-vs-rest)
        try:
            roc_auc_macro = roc_auc_score(y_true, y_proba, average='macro', multi_class='ovr')
            roc_auc_micro = roc_auc_score(y_true, y_proba, average='micro', multi_class='ovr')
        except:
            roc_auc_macro = float('nan')
            roc_auc_micro = float('nan')
        
        # Average Precision (one-vs-rest)
        try:
            # Binarize labels for average precision
            y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
            avg_precision_macro = average_precision_score(y_true_bin, y_proba, average='macro')
            avg_precision_micro = average_precision_score(y_true_bin, y_proba, average='micro')
        except:
            avg_precision_macro = float('nan')
            avg_precision_micro = float('nan')
        
        # Matthews Correlation Coefficient
        try:
            mcc = matthews_corrcoef(y_true, y_pred)
        except:
            mcc = float('nan')
        
        # Confusion Matrix (explicit labels so shape matches class_names)
        cm = confusion_matrix(y_true, y_pred, labels=class_label_indices)

        # Create per-class metrics DataFrame
        per_class_metrics = pd.DataFrame({
            'class': class_names,
            'precision': precision_per_class,
            'recall': recall_per_class,
            'f1': f1_per_class
        })
        
        metrics = {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_micro': precision_micro,
            'recall_micro': recall_micro,
            'f1_micro': f1_micro,
            'roc_auc_macro': roc_auc_macro,
            'roc_auc_micro': roc_auc_micro,
            'avg_precision_macro': avg_precision_macro,
            'avg_precision_micro': avg_precision_micro,
            'mcc': mcc,
            'confusion_matrix': cm,
            'per_class_metrics': per_class_metrics
        }
        
        return metrics

