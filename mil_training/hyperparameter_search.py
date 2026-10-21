"""
Hyperparameter search script for MIL models with random search and k-fold cross-validation.

This script performs hyperparameter optimization using random search, supporting both
train/validation split (for large datasets) and k-fold cross-validation (for smaller datasets).
After finding optimal hyperparameters, it retrains on the full training set.
"""

import torch
import pandas as pd
import numpy as np
import os
import sys
import json
import argparse
from scipy.stats import loguniform, uniform
from sklearn.model_selection import KFold
import random
from typing import Dict, List, Tuple, Any
from copy import deepcopy

# Add this package directory to Python path so `from modules import ...` works.
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import modules
from modules import (
    # Core functionality
    GatedMILClassifier, FeatureDataset, create_patient_level_dataloaders,
    # Training and evaluation
    train_model, evaluate_model,
    # Utilities
    set_seed,
    # Logging
    setup_logger, get_logger, log_args, log_section,
    # Balanced training
    BalancedTrainingConfig, analyze_class_distribution
)

import warnings
warnings.filterwarnings('ignore')


def parse_arguments():
    """Parse command line arguments for hyperparameter search."""
    parser = argparse.ArgumentParser(description='Hyperparameter search with random search and k-fold CV')
    
    # Dataset and model parameters
    parser.add_argument('--h5_directory', type=str, required=True,
                       help='Directory containing H5 feature files')
    parser.add_argument('--master_df_path', type=str, required=True,
                       help='Path to master CSV file')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for results')
    
    # Search configuration
    parser.add_argument('--search_mode', type=str, default='train_val',
                       choices=['train_val', 'k_fold'],
                       help='Search mode: train_val (train/val split) or k_fold (k-fold CV)')
    parser.add_argument('--n_trials', type=int, default=20,
                       help='Number of random search trials')
    parser.add_argument('--k_folds', type=int, default=5,
                       help='Number of folds for k-fold CV (only used if search_mode=k_fold)')
    parser.add_argument('--val_split', type=float, default=0.2,
                       help='Validation split fraction (only used if search_mode=train_val)')
    
    # Training parameters (fixed or will be searched)
    parser.add_argument('--num_epochs', type=int, default=30,
                       help='Number of training epochs per trial')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for training')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay (L2 regularization)')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'],
                       help='Device to use for training')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random seed for reproducibility')
    
    # Hyperparameter search spaces
    parser.add_argument('--learning_rate_range', type=str, default='1e-5,1e-3',
                       help='Learning rate range for log-uniform search. Can be a single value (e.g., "1e-4") '
                       'or a range (comma-separated: min,max, e.g., "1e-5,1e-3")')
    parser.add_argument('--dropout_range', type=str, default='0.0,0.5',
                       help='Dropout rate range for uniform search. Can be a single value (e.g., "0.2") '
                       'or a range (comma-separated: min,max, e.g., "0.0,0.5")')
    parser.add_argument('--attention_dropout_range', type=str, default='0.0,0.5',
                       help='Attention dropout rate range for uniform search. Can be a single value (e.g., "0.0") '
                       'or a range (comma-separated: min,max, e.g., "0.0,0.5")')
    parser.add_argument('--latent_dim_options', type=str, default='128,256,512',
                       help='Comma-separated list of latent dimensions to search (e.g., "128,256,512")')
    
    # Architecture boolean options (can be searched or fixed)
    parser.add_argument('--layer_norm_options', type=str, default=None,
                       help='Layer normalization options to search (comma-separated: "False,True"). If not provided, uses --use_layer_norm flag value.')
    parser.add_argument('--use_layer_norm', action='store_true',
                       help='Use layer normalization (fixed value if --layer_norm_options not provided)')
    parser.add_argument('--batch_norm_options', type=str, default=None,
                       help='Batch normalization options to search (comma-separated: "False,True"). If not provided, uses --use_batch_norm flag value.')
    parser.add_argument('--use_batch_norm', action='store_true',
                       help='Use batch normalization (fixed value if --batch_norm_options not provided)')
    parser.add_argument('--skip_connection_options', type=str, default=None,
                       help='Skip connection options to search (comma-separated: "False,True"). If not provided, uses --use_skip_connection flag value.')
    parser.add_argument('--use_skip_connection', action='store_true',
                       help='Use skip connections (fixed value if --skip_connection_options not provided)')
    parser.add_argument('--early_stopping_patience', type=int, default=None,
                       help='Early stopping patience (None = disabled)')
    parser.add_argument('--gradient_clip_value', type=float, default=None,
                       help='Gradient clipping value (None = disabled)')
    
    # Balanced training options (fixed during search)
    parser.add_argument('--enable_balancing', action='store_true',
                       help='Enable class balancing techniques')
    parser.add_argument('--balancing_method', type=str, default='class_weighted_loss',
                       choices=['class_weighted_loss', 'weighted_sampling', 'both'],
                       help='Method for handling class imbalance')
    parser.add_argument('--sampling_strategy', type=str, default='natural',
                       choices=['natural', 'balanced'],
                       help='Sampling strategy for training data')
    parser.add_argument('--weight_method', type=str, default='balanced',
                       choices=['inverse_frequency', 'balanced', 'effective_samples'],
                       help='Method for calculating class weights')
    
    # Best model selection
    parser.add_argument('--best_metric', type=str, default='val_auprc_macro',
                       choices=['val_loss', 'val_acc', 'val_f1_macro', 'val_f1_micro', 
                               'val_roc_auc_macro', 'val_auprc_macro', 'val_mcc', 
                               'val_sensitivity', 'val_specificity'],
                       help='Metric to use for best hyperparameter selection')
    parser.add_argument('--metric_mode', type=str, default='auto',
                       choices=['min', 'max', 'auto'],
                       help='Whether to minimize or maximize the metric (auto=infer from metric name)')
    
    # Final retraining options
    parser.add_argument('--final_retrain_epochs', type=int, default=None,
                       help='Number of epochs for final retraining (default: same as num_epochs)')
    parser.add_argument('--skip_final_retrain', action='store_true',
                       help='Skip final retraining on full dataset')
    parser.add_argument('--save_final_model', action='store_true',
                       help='Save final_model.pt at the end of each trial training (default: disabled)')
    parser.add_argument('--no_save_best_model', action='store_false', dest='save_best_model',
                       default=True, help='Disable saving best_model.pt during training (default: enabled). '
                       'Note: args.save_best_model will be False when this flag is used, True otherwise.')
    parser.add_argument('--save_training_logs_too', action='store_true',
                       help='Save detailed training logs (epoch-by-epoch) to the log file. '
                       'If not set, training logs only go to console (default: disabled)')
    
    return parser.parse_args()


def infer_metric_mode(best_metric):
    """Automatically infer whether to minimize or maximize a metric."""
    minimize_metrics = ['val_loss']
    maximize_metrics = ['val_acc', 'val_f1_macro', 'val_f1_micro', 'val_roc_auc_macro', 
                       'val_auprc_macro', 'val_mcc', 'val_sensitivity', 'val_specificity']
    
    if best_metric in minimize_metrics:
        return 'min'
    elif best_metric in maximize_metrics:
        return 'max'
    else:
        raise ValueError(f"Unknown metric mode for {best_metric}. Please specify manually.")


def parse_range(range_str: str, param_name: str) -> Tuple[float, float]:
    """
    Parse a range string into min and max values.
    
    Supports two formats:
    - Single value: "0.0" -> treated as fixed value (min=max)
    - Range: "0.0,0.5" -> min and max values
    
    Args:
        range_str: Single value or comma-separated range string
        param_name: Name of parameter for error messages
    
    Returns:
        tuple: (min_value, max_value)
    """
    try:
        parts = [x.strip() for x in range_str.split(',')]
        
        # Single value case: treat as fixed value (min = max)
        if len(parts) == 1:
            value = float(parts[0])
            return value, value
        
        # Range case: must have exactly 2 values
        if len(parts) != 2:
            raise ValueError(f"Range must be a single value or exactly 2 values separated by comma")
        
        min_val = float(parts[0])
        max_val = float(parts[1])
        
        # For fixed values (min == max), allow it
        if min_val > max_val:
            raise ValueError(f"Minimum value must be less than or equal to maximum value")
        
        return min_val, max_val
    except ValueError as e:
        raise ValueError(f"Invalid range format for {param_name}: {range_str}. Error: {e}")


def parse_boolean_options(options_str: str, param_name: str) -> List[bool]:
    """
    Parse a boolean options string (e.g., "False,True") into list of boolean values.
    
    Args:
        options_str: Comma-separated boolean options string
        param_name: Name of parameter for error messages
    
    Returns:
        list: List of boolean values
    """
    try:
        parts = [x.strip() for x in options_str.split(',')]
        if len(parts) == 0:
            raise ValueError(f"Options string cannot be empty")
        boolean_options = []
        for part in parts:
            part_lower = part.lower()
            if part_lower in ('true', '1', 'yes', 'on'):
                boolean_options.append(True)
            elif part_lower in ('false', '0', 'no', 'off'):
                boolean_options.append(False)
            else:
                raise ValueError(f"Invalid boolean value: {part}. Must be True/False or 1/0")
        return boolean_options
    except ValueError as e:
        raise ValueError(f"Invalid boolean options format for {param_name}: {options_str}. Error: {e}")


def sample_hyperparameters(args, trial_seed: int) -> Dict[str, Any]:
    """
    Sample hyperparameters from search space.
    
    Args:
        args: Parsed arguments
        trial_seed: Random seed for this trial
    
    Returns:
        dict: Sampled hyperparameters
    """
    # Set seed for reproducibility of this trial
    random.seed(trial_seed)
    np.random.seed(trial_seed)
    
    # Parse ranges
    lr_min, lr_max = parse_range(args.learning_rate_range, 'learning_rate')
    dropout_min, dropout_max = parse_range(args.dropout_range, 'dropout')
    attn_dropout_min, attn_dropout_max = parse_range(args.attention_dropout_range, 'attention_dropout')
    
    # Sample learning rate (log-uniform) - handle fixed value case
    if lr_min == lr_max:
        learning_rate = lr_min  # Fixed value, no sampling needed
    else:
        learning_rate = loguniform.rvs(lr_min, lr_max, random_state=trial_seed)
    
    # Sample dropout (uniform) - handle fixed value case
    if dropout_min == dropout_max:
        dropout = dropout_min  # Fixed value, no sampling needed
    else:
        dropout = uniform.rvs(dropout_min, dropout_max - dropout_min, random_state=trial_seed)
    
    # Sample attention dropout (uniform) - handle fixed value case
    if attn_dropout_min == attn_dropout_max:
        attention_dropout = attn_dropout_min  # Fixed value, no sampling needed
    else:
        attention_dropout = uniform.rvs(attn_dropout_min, attn_dropout_max - attn_dropout_min, 
                                       random_state=trial_seed)
    
    # Sample latent dimension (discrete choice)
    # Handle both comma-separated and space-separated formats
    # First try splitting by comma, then by space if needed
    if ',' in args.latent_dim_options:
        parts = [x.strip() for x in args.latent_dim_options.split(',') if x.strip()]
    else:
        # Fallback: space-separated
        parts = [x.strip() for x in args.latent_dim_options.split() if x.strip()]
    
    latent_dim_options = [int(x) for x in parts if x]
    if len(latent_dim_options) == 0:
        raise ValueError(f"No valid latent dimension options found in: {args.latent_dim_options}")
    if len(latent_dim_options) != len(set(latent_dim_options)):
        logger = get_logger('hyperparameter_search')
        logger.warning(f"Duplicate values in latent_dim_options, removing duplicates")
        latent_dim_options = list(set(latent_dim_options))
    latent_dim = random.choice(latent_dim_options)
    
    # Sample boolean options (if search space provided, otherwise use fixed value)
    use_layer_norm = random.choice(parse_boolean_options(args.layer_norm_options, 'layer_norm')) if args.layer_norm_options else args.use_layer_norm
    use_batch_norm = random.choice(parse_boolean_options(args.batch_norm_options, 'batch_norm')) if args.batch_norm_options else args.use_batch_norm
    use_skip_connection = random.choice(parse_boolean_options(args.skip_connection_options, 'skip_connection')) if args.skip_connection_options else args.use_skip_connection
    
    return {
        'learning_rate': float(learning_rate),
        'dropout': float(dropout),
        'attention_dropout': float(attention_dropout),
        'latent_dim': int(latent_dim),
        'use_layer_norm': bool(use_layer_norm),
        'use_batch_norm': bool(use_batch_norm),
        'use_skip_connection': bool(use_skip_connection)
    }


def run_single_fold(h5_directory, train_df, val_df, hyperparams, args, trial_idx: int, 
                   fold_idx: int) -> Dict[str, Any]:
    """
    Run training for a single fold in k-fold CV using train_model.
    
    Args:
        h5_directory: Directory with H5 feature files
        train_df: Training dataframe
        val_df: Validation dataframe  
        hyperparams: Dictionary of hyperparameters
        args: Parsed arguments
        trial_idx: Trial index
        fold_idx: Fold index
    
    Returns:
        dict: Fold results
    """
    trial_seed = args.random_state + trial_idx * 1000 + fold_idx * 100
    set_seed(trial_seed)
    
    logger = get_logger('hyperparameter_search')
    logger.info(f"Fold {fold_idx + 1}/{args.k_folds}")
    logger.info(f"Train slides: {len(train_df)}, Val slides: {len(val_df)}")
    
    # Create balanced training config
    balanced_config = BalancedTrainingConfig(
        enable_balancing=args.enable_balancing,
        balancing_method=args.balancing_method,
        sampling_strategy=args.sampling_strategy,
        weight_method=args.weight_method
    )
    
    # Determine device
    if args.device == 'auto':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    # Create temporary model save directory only if we're saving models
    # If neither save_best_model nor save_final_model is enabled, don't create the directory
    trial_model_dir = os.path.join(args.output_dir, 'search_trials', 
                                   f'trial_{trial_idx + 1}', f'fold_{fold_idx + 1}')
    # Directory will be created by train_model() if needed, so we don't create it here

    
    # Combine train and val for this fold, then use train_model with a way to specify the split
    # For simplicity, we'll create a combined df and use a custom approach
    # Actually, let's use train_model but with the train_df only, and manually evaluate on val_df
    # But train_model expects to do the splitting itself...
    
    # Simpler approach: Use train_df as the master_df and let train_model split it
    # But we want to use our pre-defined val set. So we need a different approach.
    # For now, let's use train_df only and let train_model create a small val split for monitoring
    # The actual validation will be on our val_df, but we'll use train_model's val metrics as proxy
    
    try:
        # Use train_df as master_df, train_model will create its own val split
        # We'll use a small val_split just for monitoring during training
        logger = get_logger('hyperparameter_search')
        model, training_metrics, class_names = train_model(
            h5_directory=h5_directory,
            master_df=train_df,  # Only training data
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=hyperparams['learning_rate'],
            weight_decay=args.weight_decay,
            val_split=0.1,  # Small validation set for monitoring
            model_save_dir=trial_model_dir,
            device=device,
            use_patient_level_split=True,
            random_state=trial_seed,
            balanced_config=balanced_config,
            best_metric=args.best_metric,
            metric_mode=args.metric_mode,
            latent_dim=hyperparams['latent_dim'],
            dropout=hyperparams['dropout'],
            attention_dropout=hyperparams['attention_dropout'],
            use_layer_norm=hyperparams['use_layer_norm'],
            use_batch_norm=hyperparams['use_batch_norm'],
            use_skip_connection=hyperparams['use_skip_connection'],
            early_stopping_patience=args.early_stopping_patience,
            gradient_clip_value=args.gradient_clip_value,
            save_final_model=args.save_final_model,
            save_best_model=args.save_best_model,
            logger=logger if args.save_training_logs_too else None,
            save_training_logs_too=args.save_training_logs_too
        )
        
        # Evaluate on the actual validation set (val_df)
        from modules.feature_dataset import FeatureDataset, collate_fn
        from torch.utils.data import DataLoader
        
        val_dataset = FeatureDataset(h5_directory, val_df)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                               collate_fn=collate_fn, num_workers=4)
        
        val_metrics = evaluate_model(model, val_loader, class_names, device)
        metric_key = args.best_metric.replace('val_', '')
        best_metric_value = val_metrics.get(metric_key, val_metrics.get('accuracy'))
        
        # Ensure best_metric_value is a single number, not a list
        if isinstance(best_metric_value, list):
            best_metric_value = best_metric_value[-1] if best_metric_value else (float('inf') if args.metric_mode == 'min' else float('-inf'))
        best_metric_value = float(best_metric_value) if best_metric_value is not None else (float('inf') if args.metric_mode == 'min' else float('-inf'))
        
        result = {
            'trial_idx': trial_idx,
            'fold_idx': fold_idx,
            'hyperparameters': hyperparams,
            'best_metric': args.best_metric,
            'best_metric_value': float(best_metric_value),
            'training_metrics': {k: (v[-1] if isinstance(v, list) else v) for k, v in training_metrics.items()},
            'val_metrics': {k: float(v) for k, v in val_metrics.items() if isinstance(v, (int, float, np.number))},
            'status': 'success'
        }
        
        logger = get_logger('hyperparameter_search')
        logger.info(f"✓ Fold completed. {args.best_metric}: {best_metric_value:.4f}")
        return result
        
    except Exception as e:
        logger = get_logger('hyperparameter_search')
        logger.error(f"✗ Fold failed: {str(e)}")
        return {
            'trial_idx': trial_idx,
            'fold_idx': fold_idx,
            'hyperparameters': hyperparams,
            'best_metric': args.best_metric,
            'best_metric_value': float('inf') if args.metric_mode == 'min' else float('-inf'),
            'status': 'failed',
            'error': str(e)
        }


def run_single_trial(h5_directory, master_df, hyperparams, args, trial_idx: int, 
                    fold_idx: int = None) -> Dict[str, Any]:
    """
    Run a single hyperparameter trial.
    
    Args:
        h5_directory: Directory with H5 feature files
        master_df: DataFrame with labels
        hyperparams: Dictionary of hyperparameters
        args: Parsed arguments
        trial_idx: Trial index
        fold_idx: Fold index (for k-fold CV, None for train_val)
    
    Returns:
        dict: Trial results including metrics and hyperparameters
    """
    trial_seed = args.random_state + trial_idx * 1000 + (fold_idx or 0) * 100
    
    logger = get_logger('hyperparameter_search')
    if fold_idx is not None:
        log_section(logger, f"Trial {trial_idx + 1}/{args.n_trials}, Fold {fold_idx + 1}/{args.k_folds}")
    else:
        log_section(logger, f"Trial {trial_idx + 1}/{args.n_trials}")
    logger.info("Hyperparameters:")
    for key, value in hyperparams.items():
        logger.info(f"  {key}: {value}")
    
    # Create balanced training config
    balanced_config = BalancedTrainingConfig(
        enable_balancing=args.enable_balancing,
        balancing_method=args.balancing_method,
        sampling_strategy=args.sampling_strategy,
        weight_method=args.weight_method
    )
    
    # Determine device
    if args.device == 'auto':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    # Create temporary model save directory for this trial
    # Directory will be created by train_model() if needed (based on save flags)
    if fold_idx is not None:
        trial_model_dir = os.path.join(args.output_dir, 'search_trials', 
                                       f'trial_{trial_idx + 1}', f'fold_{fold_idx + 1}')
    else:
        trial_model_dir = os.path.join(args.output_dir, 'search_trials', f'trial_{trial_idx + 1}')
    # Don't create directory here - let train_model() handle it based on save flags
    
    # Train model
    try:
        logger = get_logger('hyperparameter_search')
        model, training_metrics, class_names = train_model(
            h5_directory=h5_directory,
            master_df=master_df,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=hyperparams['learning_rate'],
            weight_decay=args.weight_decay,
            val_split=args.val_split if fold_idx is None else None,  # Only use val_split for train_val mode
            model_save_dir=trial_model_dir,
            device=device,
            use_patient_level_split=True,
            random_state=trial_seed,
            balanced_config=balanced_config,
            best_metric=args.best_metric,
            metric_mode=args.metric_mode,
            latent_dim=hyperparams['latent_dim'],
            dropout=hyperparams['dropout'],
            attention_dropout=hyperparams['attention_dropout'],
            use_layer_norm=hyperparams['use_layer_norm'],
            use_batch_norm=hyperparams['use_batch_norm'],
            use_skip_connection=hyperparams['use_skip_connection'],
            early_stopping_patience=args.early_stopping_patience,
            gradient_clip_value=args.gradient_clip_value,
            save_final_model=args.save_final_model,
            save_best_model=args.save_best_model,
            logger=logger if args.save_training_logs_too else None,
            save_training_logs_too=args.save_training_logs_too
        )
        
        # Get best metric value from training
        # Try to load from checkpoint first (if best_model.pt was saved)
        best_metric_value = None
        checkpoint_path = os.path.join(trial_model_dir, 'best_model.pt')
        
        if args.save_best_model and os.path.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                checkpoint_best = checkpoint.get('best_metric_value', None)
                if checkpoint_best is not None:
                    best_metric_value = float(checkpoint_best)
                    logger.debug(f"✓ Best metric from checkpoint: {best_metric_value:.4f}")
            except Exception as e:
                logger.warning(f"Could not load best value from checkpoint: {e}")
        
        # Fallback: extract from metrics list if checkpoint didn't work or wasn't saved
        if best_metric_value is None:
            metric_list = training_metrics.get(args.best_metric, None)
            
            if metric_list is None:
                logger.warning(f"{args.best_metric} not found in training metrics")
                best_metric_value = float('inf') if args.metric_mode == 'min' else float('-inf')
            elif isinstance(metric_list, list):
                # If it's a list, get the best value based on metric_mode
                if len(metric_list) == 0:
                    best_metric_value = float('inf') if args.metric_mode == 'min' else float('-inf')
                elif args.metric_mode == 'min':
                    best_metric_value = min(metric_list)
                else:
                    best_metric_value = max(metric_list)
                logger.debug(f"✓ Best metric from list ({args.metric_mode}): {best_metric_value:.4f} (from {len(metric_list)} epochs)")
            else:
                # It's already a single value, use it as is
                best_metric_value = float(metric_list)
                logger.debug(f"✓ Best metric (single value): {best_metric_value:.4f}")
        
        result = {
            'trial_idx': trial_idx,
            'fold_idx': fold_idx,
            'hyperparameters': hyperparams,
            'best_metric': args.best_metric,
            'best_metric_value': float(best_metric_value),
            'training_metrics': {k: (v[-1] if isinstance(v, list) else v) for k, v in training_metrics.items()},
            'status': 'success'
        }
        
        logger.info(f"✓ Trial completed. Best {args.best_metric}: {best_metric_value:.4f}")
        return result
        
    except Exception as e:
        logger = get_logger('hyperparameter_search')
        logger.error(f"✗ Trial failed: {str(e)}")
        return {
            'trial_idx': trial_idx,
            'fold_idx': fold_idx,
            'hyperparameters': hyperparams,
            'best_metric': args.best_metric,
            'best_metric_value': float('inf') if args.metric_mode == 'min' else float('-inf'),
            'status': 'failed',
            'error': str(e)
        }


def create_k_fold_patient_splits(master_df, k_folds, random_state):
    """
    Create k-fold patient-level splits for cross-validation.
    
    Args:
        master_df: DataFrame with MRN and diagnosis columns
        k_folds: Number of folds
        random_state: Random seed
    
    Returns:
        list: List of (train_patients, val_patients) tuples
    """
    # Get all unique patients
    all_patients = master_df['MRN'].unique()
    np.random.seed(random_state)
    np.random.shuffle(all_patients)
    
    # Create k-fold splits
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)
    splits = []
    
    for train_idx, val_idx in kf.split(all_patients):
        train_patients = all_patients[train_idx]
        val_patients = all_patients[val_idx]
        splits.append((train_patients, val_patients))
    
    return splits


def run_k_fold_trial(h5_directory, master_df, hyperparams, args, trial_idx: int) -> Dict[str, Any]:
    """
    Run a single trial with k-fold cross-validation.
    
    Args:
        h5_directory: Directory with H5 feature files
        master_df: DataFrame with labels
        hyperparams: Dictionary of hyperparameters
        args: Parsed arguments
        trial_idx: Trial index
    
    Returns:
        dict: Trial results with average metrics across folds
    """
    # Create k-fold patient splits
    trial_seed = args.random_state + trial_idx * 1000
    patient_splits = create_k_fold_patient_splits(master_df, args.k_folds, trial_seed)
    
    fold_results = []
    
    for fold_idx, (train_patients, val_patients) in enumerate(patient_splits):
        # Create fold-specific master_df
        train_slides = master_df[master_df['MRN'].isin(train_patients)]['slide_name'].tolist()
        val_slides = master_df[master_df['MRN'].isin(val_patients)]['slide_name'].tolist()
        
        # Create train and val dataframes
        train_df = master_df[master_df['slide_name'].isin(train_slides)].copy()
        val_df = master_df[master_df['slide_name'].isin(val_slides)].copy()
        
        # Run training for this fold
        fold_result = run_single_fold(h5_directory, train_df, val_df, hyperparams, args, 
                                     trial_idx, fold_idx)
        fold_results.append(fold_result)
    
    # Aggregate results across folds
    successful_folds = [r for r in fold_results if r['status'] == 'success']
    
    if len(successful_folds) == 0:
        return {
            'trial_idx': trial_idx,
            'hyperparameters': hyperparams,
            'best_metric': args.best_metric,
            'best_metric_value': float('inf') if args.metric_mode == 'min' else float('-inf'),
            'status': 'failed',
            'fold_results': fold_results
        }
    
    # Calculate average metric value across folds
    avg_metric_value = np.mean([r['best_metric_value'] for r in successful_folds])
    std_metric_value = np.std([r['best_metric_value'] for r in successful_folds])
    
    # Aggregate training metrics
    if len(successful_folds) > 0 and 'training_metrics' in successful_folds[0]:
        metric_keys = successful_folds[0]['training_metrics'].keys()
        avg_training_metrics = {}
        for key in metric_keys:
            values = [r['training_metrics'][key] for r in successful_folds if key in r.get('training_metrics', {})]
            if values:
                avg_training_metrics[key] = float(np.mean(values))
    else:
        avg_training_metrics = {}
    
    logger = get_logger('hyperparameter_search')
    logger.info(f"✓ K-fold trial completed. Average {args.best_metric}: {avg_metric_value:.4f} ± {std_metric_value:.4f}")
    
    return {
        'trial_idx': trial_idx,
        'hyperparameters': hyperparams,
        'best_metric': args.best_metric,
        'best_metric_value': float(avg_metric_value),
        'best_metric_std': float(std_metric_value),
        'training_metrics': avg_training_metrics,
        'fold_results': fold_results,
        'n_successful_folds': len(successful_folds),
        'status': 'success'
    }


def find_best_hyperparameters(all_trials: List[Dict], metric_mode: str) -> Dict:
    """
    Find the best hyperparameters from all trials.
    
    Args:
        all_trials: List of trial results
        metric_mode: 'min' or 'max'
    
    Returns:
        dict: Best trial result
    """
    successful_trials = [t for t in all_trials if t['status'] == 'success']
    
    if len(successful_trials) == 0:
        raise ValueError("No successful trials found!")
    
    if metric_mode == 'min':
        best_trial = min(successful_trials, key=lambda x: x['best_metric_value'])
    else:
        best_trial = max(successful_trials, key=lambda x: x['best_metric_value'])
    
    return best_trial


def retrain_with_best_hyperparameters(h5_directory, master_df, best_hyperparams, args, logger=None):
    """
    Retrain model on full training set with best hyperparameters.
    
    Args:
        h5_directory: Directory with H5 feature files
        master_df: DataFrame with labels
        best_hyperparams: Best hyperparameters found
        args: Parsed arguments
        logger: Logger instance (optional, will get if not provided)
    """
    if logger is None:
        logger = get_logger('hyperparameter_search')
    
    log_section(logger, "FINAL RETRAINING WITH BEST HYPERPARAMETERS")
    logger.info("Retraining on full training set with best hyperparameters:")
    for key, value in best_hyperparams.items():
        logger.info(f"  {key}: {value}")
    
    # Create balanced training config
    balanced_config = BalancedTrainingConfig(
        enable_balancing=args.enable_balancing,
        balancing_method=args.balancing_method,
        sampling_strategy=args.sampling_strategy,
        weight_method=args.weight_method
    )
    
    # Determine device
    if args.device == 'auto':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    # Final model save directory
    final_model_dir = os.path.join(args.output_dir, 'models')
    os.makedirs(final_model_dir, exist_ok=True)
    
    # Number of epochs for final training
    final_epochs = args.final_retrain_epochs if args.final_retrain_epochs else args.num_epochs
    
    # Retrain on full dataset (use all data for training, no validation split)
    # We'll use a very small val_split just for monitoring, but the model will use all data
    # Always save best_model.pt for final retraining (this is the model we want to use)
    model, training_metrics, class_names = train_model(
        h5_directory=h5_directory,
        master_df=master_df,
        num_epochs=final_epochs,
        batch_size=args.batch_size,
        learning_rate=best_hyperparams['learning_rate'],
        weight_decay=args.weight_decay,
        val_split=0.1,  # Small validation set just for monitoring
        model_save_dir=final_model_dir,
        device=device,
        use_patient_level_split=True,
        random_state=args.random_state,
        balanced_config=balanced_config,
        best_metric=args.best_metric,
        metric_mode=args.metric_mode,
        latent_dim=best_hyperparams['latent_dim'],
        dropout=best_hyperparams['dropout'],
        attention_dropout=best_hyperparams['attention_dropout'],
        use_layer_norm=best_hyperparams['use_layer_norm'],
        use_batch_norm=best_hyperparams['use_batch_norm'],
        use_skip_connection=best_hyperparams['use_skip_connection'],
        early_stopping_patience=args.early_stopping_patience,
        gradient_clip_value=args.gradient_clip_value,
        save_final_model=args.save_final_model,
        save_best_model=True,  # Always save best model for final retraining
        logger=logger if args.save_training_logs_too else None,
        save_training_logs_too=args.save_training_logs_too
    )
    
    # Verify best_model.pt was saved
    best_model_path = os.path.join(final_model_dir, 'best_model.pt')
    if os.path.exists(best_model_path):
        logger.info(f"✓ Final best model saved to: {best_model_path}")
    else:
        logger.warning(f"best_model.pt not found at {best_model_path}")
    
    return model, training_metrics, class_names


def save_search_results(all_trials: List[Dict], best_trial: Dict, args, output_dir):
    """
    Save hyperparameter search results to JSON.
    
    Args:
        all_trials: List of all trial results
        best_trial: Best trial result
        args: Parsed arguments
        output_dir: Output directory
    """
    results = {
        'search_configuration': {
            'search_mode': args.search_mode,
            'n_trials': args.n_trials,
            'k_folds': args.k_folds if args.search_mode == 'k_fold' else None,
            'val_split': args.val_split if args.search_mode == 'train_val' else None,
            'best_metric': args.best_metric,
            'metric_mode': args.metric_mode,
            'num_epochs_per_trial': args.num_epochs
        },
        'search_space': {
            'learning_rate': {'range': args.learning_rate_range, 'distribution': 'log_uniform'},
            'dropout': {'range': args.dropout_range, 'distribution': 'uniform'},
            'attention_dropout': {'range': args.attention_dropout_range, 'distribution': 'uniform'},
            'latent_dim': {'options': args.latent_dim_options.split(',')},
            'use_layer_norm': {'options': args.layer_norm_options.split(',') if args.layer_norm_options else [str(args.use_layer_norm)], 'searched': args.layer_norm_options is not None},
            'use_batch_norm': {'options': args.batch_norm_options.split(',') if args.batch_norm_options else [str(args.use_batch_norm)], 'searched': args.batch_norm_options is not None},
            'use_skip_connection': {'options': args.skip_connection_options.split(',') if args.skip_connection_options else [str(args.use_skip_connection)], 'searched': args.skip_connection_options is not None}
        },
        'all_trials': all_trials,
        'best_trial': best_trial,
        'summary': {
            'total_trials': len(all_trials),
            'successful_trials': len([t for t in all_trials if t['status'] == 'success']),
            'failed_trials': len([t for t in all_trials if t['status'] == 'failed']),
            'best_metric_value': best_trial['best_metric_value'],
            'best_hyperparameters': best_trial['hyperparameters']
        }
    }
    
    results_path = os.path.join(output_dir, 'hyperparameter_search_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger = get_logger('hyperparameter_search')
    logger.info(f"✓ Search results saved to: {results_path}")


def main():
    """Main hyperparameter search function."""
    args = parse_arguments()
    
    # Set up logging
    logger = setup_logger('hyperparameter_search', args.output_dir)
    
    # Set seed for reproducibility
    set_seed(args.random_state)
    
    log_section(logger, "HYPERPARAMETER SEARCH")
    logger.info(f"Search mode: {args.search_mode.upper()}")
    logger.info(f"Number of trials: {args.n_trials}")
    if args.search_mode == 'k_fold':
        logger.info(f"K-folds: {args.k_folds}")
    else:
        logger.info(f"Validation split: {args.val_split}")
    
    # Log arguments
    log_args(logger, args, "Hyperparameter Search")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load master dataframe
    master_df = pd.read_csv(args.master_df_path)
    logger.info(f"Dataset loaded: {len(master_df)} slides")
    logger.info(f"Unique patients: {master_df['MRN'].nunique()}")
    
    # Log search space configuration
    log_section(logger, "HYPERPARAMETER SEARCH SPACE")
    logger.info(f"Learning rate range: {args.learning_rate_range} (log-uniform)")
    logger.info(f"Dropout range: {args.dropout_range} (uniform)")
    logger.info(f"Attention dropout range: {args.attention_dropout_range} (uniform)")
    # Parse and display latent dim options to verify they're correct
    if ',' in args.latent_dim_options:
        parts = [x.strip() for x in args.latent_dim_options.split(',') if x.strip()]
    else:
        parts = [x.strip() for x in args.latent_dim_options.split() if x.strip()]
    parsed_latent_dims = [int(x) for x in parts if x]
    logger.info(f"Latent dimension options: {args.latent_dim_options} -> {parsed_latent_dims}")
    if args.layer_norm_options:
        logger.info(f"Layer norm options: {args.layer_norm_options}")
    else:
        logger.info(f"Layer norm: {args.use_layer_norm} (fixed)")
    if args.batch_norm_options:
        logger.info(f"Batch norm options: {args.batch_norm_options}")
    else:
        logger.info(f"Batch norm: {args.use_batch_norm} (fixed)")
    if args.skip_connection_options:
        logger.info(f"Skip connection options: {args.skip_connection_options}")
    else:
        logger.info(f"Skip connection: {args.use_skip_connection} (fixed)")
    
    # Infer metric mode if auto
    metric_mode = infer_metric_mode(args.best_metric) if args.metric_mode == 'auto' else args.metric_mode
    
    # Run hyperparameter search
    all_trials = []
    
    for trial_idx in range(args.n_trials):
        # Sample hyperparameters
        trial_seed = args.random_state + trial_idx
        hyperparams = sample_hyperparameters(args, trial_seed)
        
        # Run trial
        if args.search_mode == 'k_fold':
            trial_result = run_k_fold_trial(args.h5_directory, master_df, hyperparams, args, trial_idx)
        else:
            trial_result = run_single_trial(args.h5_directory, master_df, hyperparams, args, trial_idx)
        
        all_trials.append(trial_result)
    
    # Find best hyperparameters
    log_section(logger, "SEARCH COMPLETE - FINDING BEST HYPERPARAMETERS")
    
    best_trial = find_best_hyperparameters(all_trials, metric_mode)
    
    logger.info(f"Best trial: {best_trial['trial_idx'] + 1}")
    logger.info(f"Best {args.best_metric}: {best_trial['best_metric_value']:.4f}")
    logger.info("Best hyperparameters:")
    for key, value in best_trial['hyperparameters'].items():
        logger.info(f"  {key}: {value}")
    
    # Save search results
    save_search_results(all_trials, best_trial, args, args.output_dir)
    
    # Final retraining with best hyperparameters
    if not args.skip_final_retrain:
        retrain_with_best_hyperparameters(
            args.h5_directory, 
            master_df, 
            best_trial['hyperparameters'], 
            args,
            logger
        )
    else:
        logger.info("⏭️  Skipping final retraining (--skip_final_retrain specified)")
    
    # Final summary
    log_section(logger, "HYPERPARAMETER SEARCH COMPLETED SUCCESSFULLY!")
    logger.info(f"✓ Completed {args.n_trials} trials")
    logger.info(f"✓ Successful trials: {len([t for t in all_trials if t['status'] == 'success'])}")
    logger.info(f"✓ Best {args.best_metric}: {best_trial['best_metric_value']:.4f}")
    logger.info(f"✓ Results saved to: {args.output_dir}")
    if not args.skip_final_retrain:
        logger.info(f"✓ Final model saved to: {os.path.join(args.output_dir, 'models')}")


if __name__ == "__main__":
    main()

