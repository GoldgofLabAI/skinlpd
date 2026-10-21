"""
Training and evaluation utilities for ML models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
import numpy as np
import os
import time
import logging
import sys
from sklearn.metrics import accuracy_score

from .gated_attn_pool import GatedMILClassifier
from .feature_dataset import FeatureDataset, create_patient_level_dataloaders, collate_fn
from .metrics import calculate_comprehensive_metrics
from .balanced_training import BalancedTrainingConfig, analyze_class_distribution
from .logger import get_logger


def train_model(h5_directory, master_df, num_epochs=50, batch_size=16, learning_rate=0.001, 
                weight_decay=0.01, val_split=0.2, model_save_dir='models', device=None, 
                use_patient_level_split=True, random_state=42, balanced_config=None,
                best_metric='val_loss', metric_mode='min', latent_dim=128, dropout=0.0,
                attention_dropout=0.0, use_layer_norm=False, use_batch_norm=False,
                use_skip_connection=False, early_stopping_patience=None, gradient_clip_value=None,
                save_final_model=False, save_best_model=True, logger=None, save_training_logs_too=False):
    """
    Train the GatedMILClassifier for multi-class classification with patient-level splitting.
    
    Args:
        h5_directory (str): Directory containing H5 files with extracted features
        master_df (pd.DataFrame): DataFrame containing diagnosis labels, slide_name, and MRN columns
        num_epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        learning_rate (float): Learning rate for optimizer
        weight_decay (float): L2 regularization strength (weight decay)
        val_split (float): Fraction of patients to use for validation
        model_save_dir (str): Directory to save trained models
        device (torch.device): Device to use for training
        use_patient_level_split (bool): Whether to use patient-level splitting (recommended)
        random_state (int): Random seed for reproducible splits
        balanced_config (BalancedTrainingConfig): Configuration for balanced training (optional)
        best_metric (str): Metric to use for best model selection 
            Options: 'val_loss', 'val_acc', 'val_f1_macro', 'val_f1_micro', 'val_roc_auc_macro', 
                    'val_auprc_macro', 'val_mcc', 'val_sensitivity', 'val_specificity'
        metric_mode (str): 'min' for metrics to minimize (loss), 'max' for metrics to maximize (accuracy, f1, etc.)
        latent_dim (int): Latent dimension for attention mechanism (smaller = less overfitting)
        dropout (float): Dropout rate for regularization (0.0-0.5 recommended)
        attention_dropout (float): Dropout rate specifically for attention scores
        use_layer_norm (bool): Whether to use layer normalization for training stability
        use_batch_norm (bool): Whether to use batch normalization (stronger regularization than layer norm)
        use_skip_connection (bool): Whether to use skip connections in attention
        early_stopping_patience (int): Early stopping patience (None = disabled)
        gradient_clip_value (float): Gradient clipping value (None = disabled)
        save_final_model (bool): Whether to save final_model.pt at the end of training (default: False)
        save_best_model (bool): Whether to save best_model.pt during training (default: True)
        logger (logging.Logger): Optional logger instance to use for training logs. If provided and 
            save_training_logs_too=True, training logs will be saved to the same log file as the calling script.
        save_training_logs_too (bool): If True and logger is provided, detailed training logs (epoch-by-epoch)
            will be saved to the log file. If False, training logs only go to console (default: False)
    
    Returns:
        tuple: Trained model, training metrics, and class names
    """
    # Get logger - use provided logger if save_training_logs_too is True, otherwise use default training logger
    if save_training_logs_too and logger is not None:
        # Use the provided logger which has file output configured
        # The logger should already have both console and file handlers from setup_logger
        pass  # Keep using the provided logger
    elif logger is not None:
        # Logger provided but save_training_logs_too is False - use it anyway (it has console output)
        pass  # Keep using the provided logger
    else:
        # No logger provided - create a simple logger for console output only
        logger = get_logger('training')
        # Ensure it has at least a console handler if it doesn't have any handlers
        if not logger.handlers:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            logger.setLevel(logging.INFO)
    
    # Log which logger is being used (for debugging)
    if save_training_logs_too:
        logger.info(f"Training logs will be saved to file (save_training_logs_too=True)")
    else:
        logger.info(f"Training logs will only appear in console (save_training_logs_too=False)")
    
    # Set device
    if device is None:
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create model save directory only if we're saving any models
    if save_final_model or save_best_model:
        os.makedirs(model_save_dir, exist_ok=True)
    
    # Get class information from the dataset
    unique_diagnoses = sorted(master_df['diagnosis'].unique())
    num_classes = len(unique_diagnoses)
    class_names = unique_diagnoses
    
    # Determine classification type
    is_binary = num_classes == 2
    classification_type = "Binary" if is_binary else "Multi-class"
    
    logger.info(f"{classification_type} classification setup:")
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Classes: {class_names}")
    
    if is_binary:
        logger.info(f"Binary classification detected - will generate ROC curve and additional binary metrics")
    
    # Initialize balanced training configuration if not provided
    if balanced_config is None:
        balanced_config = BalancedTrainingConfig(enable_balancing=False)
    
    # Print balanced training configuration
    balanced_config.print_config()
    
    # Create patient-level dataloaders (no test set)
    if use_patient_level_split:
        logger.info("Using patient-level splitting to prevent data leakage...")
        train_loader, val_loader, _ = create_patient_level_dataloaders(
            h5_directory=h5_directory,
            master_df=master_df,
            batch_size=batch_size,
            num_workers=4,
            val_split=val_split,
            test_split=0.0,  # No test set - only train and validation
            random_state=random_state,
            stratify_by_slide_distribution=True,
            max_iterations=100
        )
    else:
        # Fallback to old slide-level splitting (not recommended)
        logger.warning("Using slide-level splitting - may cause data leakage!")
        dataset = FeatureDataset(h5_directory, master_df)
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], 
                                                 generator=torch.Generator().manual_seed(random_state))
        
        # Get balanced sampler if needed
        train_sampler = None
        if balanced_config.enable_balancing:
            train_labels = [train_dataset.dataset.label_dict[train_dataset.dataset.slide_names[i]] 
                          for i in train_dataset.indices]
            train_sampler = balanced_config.get_sampler(train_labels)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                shuffle=(train_sampler is None), sampler=train_sampler, 
                                collate_fn=collate_fn, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.info(f"Validation samples: {len(val_loader.dataset)}")
    
    # Analyze class distribution for training set
    logger.info("Analyzing training set class distribution...")
    if use_patient_level_split:
        # For patient-level split, we need to extract labels from the master_df filtered by training patients
        # This is more complex, so let's use a simplified approach for the class distribution analysis
        train_diagnoses = master_df['diagnosis'].tolist()
        train_labels = [class_names.index(diagnosis) for diagnosis in train_diagnoses]
    else:
        train_labels = [train_loader.dataset.dataset.label_dict[train_loader.dataset.dataset.slide_names[i]] 
                       for i in train_loader.dataset.indices]
    
    class_distribution = analyze_class_distribution(train_labels, class_names)
    
    # Calculate class counts for loss function
    unique_classes, class_counts = np.unique(train_labels, return_counts=True)
    class_count_dict = {cls: count for cls, count in zip(unique_classes, class_counts)}
    
    # Initialize model with regularization parameters
    model = GatedMILClassifier(
        d_model=1024, 
        num_classes=num_classes, 
        latent_dim=latent_dim,
        dropout=dropout,
        attention_dropout=attention_dropout,
        use_layer_norm=use_layer_norm,
        use_batch_norm=use_batch_norm,
        use_skip_connection=use_skip_connection
    )
    model = model.to(device)
    
    # Print model configuration
    logger.info("Model Configuration:")
    logger.info(f"  Architecture: GatedMILClassifier")
    logger.info(f"  Input dimension: 1024")
    logger.info(f"  Latent dimension: {latent_dim}")
    logger.info(f"  Output classes: {num_classes}")
    logger.info(f"  Dropout: {dropout}")
    logger.info(f"  Attention dropout: {attention_dropout}")
    logger.info(f"  Layer normalization: {use_layer_norm}")
    logger.info(f"  Batch normalization: {use_batch_norm}")
    logger.info(f"  Skip connections: {use_skip_connection}")
    if use_batch_norm and use_layer_norm:
        logger.warning(f"Both LayerNorm and BatchNorm enabled - using LayerNorm")
    elif use_batch_norm:
        logger.info(f"✅ BatchNorm can provide strong regularization for overfitting")
        logger.info(f"📋 Note: Using drop_last=True to prevent single-sample batches")
        logger.info(f"💡 If you have very small datasets, consider LayerNorm instead")
    if dropout > 0.3:
        logger.info(f"-> Using deeper classifier due to high dropout")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    
    # Define loss function and optimizer
    criterion = balanced_config.get_loss_function(class_count_dict, num_classes, device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    logger.info(f"Loss function: {type(criterion).__name__}")
    if hasattr(criterion, 'weight') and criterion.weight is not None:
        logger.info(f"Class weights: {criterion.weight.cpu().numpy()}")
        if is_binary:
            logger.info(f"  -> For binary: negative_weight={criterion.weight[0]:.4f}, positive_weight={criterion.weight[1]:.4f}")
    elif hasattr(criterion, 'pos_weight') and criterion.pos_weight is not None:
        logger.info(f"Positive class weight: {criterion.pos_weight.cpu().numpy()}")
    else:
        logger.info("No class weighting applied")
    
    logger.info(f"Optimizer: Adam with learning rate {learning_rate} and weight decay (L2) {weight_decay}")
    
    # Training metrics - enhanced with AUPRC and additional metrics
    metrics = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'val_precision_macro': [],
        'val_recall_macro': [],
        'val_f1_macro': [],
        'val_f1_micro': [],
        'val_roc_auc_macro': [],
        'val_roc_auc_micro': [],
        'val_auprc_macro': [],  # Average Precision (AUPRC)
        'val_auprc_micro': [],  # AUPRC micro average
        'val_mcc': []
    }
    
    # Add binary-specific metrics if binary classification
    if is_binary:
        metrics.update({
            'val_sensitivity': [],
            'val_specificity': []
        })
    
    # Validate best_metric parameter
    valid_metrics = list(metrics.keys())
    if best_metric not in valid_metrics:
        raise ValueError(f"best_metric '{best_metric}' not valid. Choose from: {valid_metrics}")
    
    # Set up best metric tracking
    if metric_mode == 'min':
        best_metric_value = float('inf')
        is_better = lambda new_val, best_val: new_val < best_val
    elif metric_mode == 'max':
        best_metric_value = float('-inf')
        is_better = lambda new_val, best_val: new_val > best_val
    else:
        raise ValueError("metric_mode must be 'min' or 'max'")
    
    logger.info(f"Best model selection: {best_metric} ({metric_mode}imize)")
    logger.info(f"Binary classification: {is_binary}")
    
    # Early stopping setup
    if early_stopping_patience is not None:
        logger.info(f"Early stopping: {early_stopping_patience} epochs patience")
        early_stopping_counter = 0
        early_stopping_best = best_metric_value
    else:
        logger.info("Early stopping: Disabled")
    
    # Gradient clipping setup
    if gradient_clip_value is not None:
        logger.info(f"Gradient clipping: {gradient_clip_value}")
    else:
        logger.info("Gradient clipping: Disabled")
    
    best_model_path = os.path.join(model_save_dir, 'best_model.pt')
    
    # Training loop
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []
        
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device).long()  # Use long for CrossEntropyLoss
            
            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping if enabled
            if gradient_clip_value is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_value)
            
            optimizer.step()
            
            train_loss += loss.item()
            
            # Convert outputs to predictions
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            train_preds.extend(preds)
            train_labels.extend(labels.cpu().numpy())
        
        # Calculate training metrics
        train_loss /= len(train_loader)
        train_acc = accuracy_score(train_labels, train_preds)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        val_probs = []
        
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device)
                labels = labels.to(device).long()  # Use long for CrossEntropyLoss
                
                # Forward pass
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                
                # Get probabilities and predictions
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                
                val_probs.extend(probs)
                val_preds.extend(preds)
                val_labels.extend(labels.cpu().numpy())
        
        # Calculate validation metrics
        val_loss /= len(val_loader)
        val_probs = np.array(val_probs)
        
        # Calculate comprehensive metrics
        val_metrics = calculate_comprehensive_metrics(val_labels, val_preds, val_probs, class_names)
        
        # Save metrics - enhanced with additional metrics
        metrics['train_loss'].append(train_loss)
        metrics['val_loss'].append(val_loss)
        metrics['train_acc'].append(train_acc)
        metrics['val_acc'].append(val_metrics['accuracy'])
        metrics['val_precision_macro'].append(val_metrics['precision_macro'])
        metrics['val_recall_macro'].append(val_metrics['recall_macro'])
        metrics['val_f1_macro'].append(val_metrics['f1_macro'])
        metrics['val_roc_auc_macro'].append(val_metrics['roc_auc_macro'])
        metrics['val_auprc_macro'].append(val_metrics['avg_precision_macro'])  # AUPRC
        metrics['val_mcc'].append(val_metrics['mcc'])
        
        # Add multiclass-specific metrics if not binary
        if not is_binary:
            metrics['val_f1_micro'].append(val_metrics['f1_micro'])
            metrics['val_roc_auc_micro'].append(val_metrics['roc_auc_micro'])
            metrics['val_auprc_micro'].append(val_metrics['avg_precision_micro'])
        else:
            # For binary, micro = macro, so use macro values
            metrics['val_f1_micro'].append(val_metrics['f1_macro'])
            metrics['val_roc_auc_micro'].append(val_metrics['roc_auc_macro'])
            metrics['val_auprc_micro'].append(val_metrics['avg_precision_macro'])
            
            # Add binary-specific metrics
            metrics['val_sensitivity'].append(val_metrics.get('sensitivity', val_metrics['recall_macro']))
            metrics['val_specificity'].append(val_metrics.get('specificity', 0.0))
        
        # Save best model based on selected metric
        current_metric_value = metrics[best_metric][-1]  # Get the latest value
        model_improved = is_better(current_metric_value, best_metric_value)
        
        if model_improved:
            best_metric_value = current_metric_value
            if early_stopping_patience is not None:
                early_stopping_counter = 0
                early_stopping_best = current_metric_value
            
            # Save best model if requested
            if save_best_model:
                save_dict = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'best_metric': best_metric,
                    'best_metric_value': best_metric_value,
                    'metric_mode': metric_mode,
                    'metrics': metrics,
                    'class_names': class_names,
                    'num_classes': num_classes,
                    'model_config': {
                        'latent_dim': latent_dim,
                        'dropout': dropout,
                        'attention_dropout': attention_dropout,
                        'use_layer_norm': use_layer_norm,
                        'use_batch_norm': use_batch_norm,
                        'use_skip_connection': use_skip_connection
                    }
                }
                
                # For binary classification, save validation predictions and probabilities for ROC curve
                if num_classes == 2:
                    save_dict['best_val_labels'] = val_labels
                    save_dict['best_val_preds'] = val_preds
                    save_dict['best_val_probs'] = val_probs
                    save_dict['is_binary'] = True
                else:
                    save_dict['is_binary'] = False
                
                torch.save(save_dict, best_model_path)
                logger.info(f"🎯 New best {best_metric}: {best_metric_value:.4f} - Model saved!")
            else:
                logger.info(f"🎯 New best {best_metric}: {best_metric_value:.4f} (not saving model)")
        else:
            # Check early stopping
            if early_stopping_patience is not None:
                early_stopping_counter += 1
                logger.info(f"No improvement for {early_stopping_counter}/{early_stopping_patience} epochs")
        
        epoch_time = time.time() - start_time
        
        # Print epoch results
        logger.info(f"Epoch {epoch+1}/{num_epochs} | Time: {epoch_time:.2f}s")
        logger.info(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")
        
        # Combine validation metrics into a single line
        if is_binary:
            # Binary classification: show macro metrics + sensitivity/specificity
            sensitivity = val_metrics.get('sensitivity', val_metrics['recall_macro'])
            specificity = val_metrics.get('specificity', 0.0)
            logger.info(f"Val Metrics | Prec_M: {val_metrics['precision_macro']:.4f} | Rec_M: {val_metrics['recall_macro']:.4f} | "
                       f"F1_M: {val_metrics['f1_macro']:.4f} | ROC_AUC_M: {val_metrics['roc_auc_macro']:.4f} | "
                       f"AUPRC_M: {val_metrics['avg_precision_macro']:.4f} | MCC: {val_metrics['mcc']:.4f} | "
                       f"Sens: {sensitivity:.4f} | Spec: {specificity:.4f}")
        else:
            # Multiclass: show both macro and micro metrics
            logger.info(f"Val Metrics | Prec_M: {val_metrics['precision_macro']:.4f} | Rec_M: {val_metrics['recall_macro']:.4f} | "
                       f"F1_M: {val_metrics['f1_macro']:.4f} | ROC_AUC_M: {val_metrics['roc_auc_macro']:.4f} | "
                       f"AUPRC_M: {val_metrics['avg_precision_macro']:.4f} | MCC: {val_metrics['mcc']:.4f} | "
                       f"F1_μ: {val_metrics['f1_micro']:.4f} | ROC_AUC_μ: {val_metrics['roc_auc_micro']:.4f} | "
                       f"AUPRC_μ: {val_metrics['avg_precision_micro']:.4f}")
        
        # Highlight the metric being used for best model selection
        logger.info(f"📊 {best_metric.upper()}: {current_metric_value:.4f} {'📈' if metric_mode == 'max' else '📉'}")
        
        # Show confusion matrix for validation set
        logger.info("Confusion Matrix (Validation):")
        logger.info(f"\n{val_metrics['confusion_matrix']}")
        
        # Check early stopping
        if early_stopping_patience is not None and early_stopping_counter >= early_stopping_patience:
            logger.info(f"\n🛑 Early stopping triggered after {early_stopping_patience} epochs without improvement")
            logger.info(f"Best {best_metric}: {early_stopping_best:.4f}")
            break
    
    # Save final model if requested
    if save_final_model:
        final_model_path = os.path.join(model_save_dir, 'final_model.pt')
        torch.save({
            'epoch': num_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'class_names': class_names,
            'num_classes': num_classes
        }, final_model_path)
        print(f"Saved final model to {final_model_path}")
    
    return model, metrics, class_names


def evaluate_model(model, dataloader, class_names, device=None):
    """
    Evaluate the trained model on a dataset with comprehensive metrics for both binary and multi-class.
    
    Args:
        model (nn.Module): Trained model
        dataloader (DataLoader): DataLoader for test dataset
        class_names (list): List of class names
        device (torch.device): Device to use for evaluation
    
    Returns:
        dict: Dictionary of comprehensive evaluation metrics, including probabilities for binary classification
    """
    if device is None:
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device).long()
            
            # Forward pass
            outputs = model(features)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate comprehensive metrics
    all_probs = np.array(all_probs)
    metrics = calculate_comprehensive_metrics(all_labels, all_preds, all_probs, class_names)
    
    # Add raw predictions and probabilities for further analysis (especially ROC curves)
    metrics['raw_labels'] = all_labels
    metrics['raw_predictions'] = all_preds
    metrics['raw_probabilities'] = all_probs
    
    return metrics

