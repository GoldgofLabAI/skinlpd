"""
Enhanced training script with configurable class balancing options.

This script demonstrates how to use the balanced training features for handling class imbalance.
You can easily enable/disable different balancing strategies based on your dataset's needs.
"""

import torch
import pandas as pd
import numpy as np
import os
import sys
import json
import argparse
from sklearn.metrics import roc_curve, precision_recall_curve

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
    # Metrics and visualization
    plot_confusion_matrix, plot_per_class_metrics, plot_training_metrics, 
    plot_roc_curve, plot_precision_recall_curve,
    # Utilities
    set_seed, to_json_serializable,
    # Logging
    setup_logger, get_logger, log_args, log_section,
    # Balanced training
    BalancedTrainingConfig, analyze_class_distribution,
    # Threshold optimization
    find_optimal_threshold
)

import warnings
warnings.filterwarnings('ignore')


def parse_arguments():
    """Parse command line arguments for balanced training options."""
    parser = argparse.ArgumentParser(description='Train with configurable class balancing')
    
    # Dataset and model parameters
    parser.add_argument('--h5_directory', type=str, required=True,
                       help='Directory containing H5 feature files')
    parser.add_argument('--master_df_path', type=str, required=True,
                       help='Path to master CSV file')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for results')
    
    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for training')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split fraction')
    parser.add_argument(
        '--test_split',
        type=float,
        default=0.0,
        help='Fraction of patients to reserve as a held-out test set (patient-level split). '
             'When > 0, a test CSV mirroring the input master CSV schema will be saved for external evaluation.'
    )
    parser.add_argument('--random_state', type=int, default=42, help='Random seed')
    
    # Balanced training options
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
    parser.add_argument('--pos_weight', type=float, default=None,
                       help='Specific positive class weight for binary classification')
    parser.add_argument('--smooth_weights', type=float, default=0.0,
                       help='Smoothing factor for class weights')
    
    # Best model selection options
    parser.add_argument('--best_metric', type=str, default='val_loss',
                       choices=['val_loss', 'val_acc', 'val_f1_macro', 'val_f1_micro', 
                               'val_roc_auc_macro', 'val_auprc_macro', 'val_mcc', 
                               'val_sensitivity', 'val_specificity'],
                       help='Metric to use for best model selection')
    parser.add_argument('--metric_mode', type=str, default='auto',
                       choices=['min', 'max', 'auto'],
                       help='Whether to minimize or maximize the metric (auto=infer from metric name)')
    
    # Operating point selection
    parser.add_argument('--operating_point_selection', type=str, default='default',
                       choices=['default', 'youden', 'f1', 'sensitivity_95'],
                       help="Method for selecting optimal operating point threshold (default=0.5, youden=Youden's J, f1=F1-max, sensitivity_95=95%% sensitivity)")
    parser.add_argument('--recompute_operating_point', action='store_true',
                       help='Recompute operating point threshold on the validation set using an existing best_model.pt without retraining')
    parser.add_argument('--eval_only', action='store_true',
                       help='Load existing best_model.pt and run validation evaluation + save results without retraining')
    parser.add_argument('--case_level', action='store_true',
                       help='When optimizing operating point, work at case-part level: aggregate slide probs by (case_id, part) before threshold search')
    parser.add_argument('--case_level_method', type=str, default='mean',
                       choices=['mean', 'max'],
                       help='How to aggregate slide probabilities per case-part when selecting operating point (default: mean)')
    parser.add_argument('--case_id_column', type=str, default='sp_specimen_id',
                       help='CSV column for case/specimen ID used for case-level operating point selection (default: sp_specimen_id)')
    parser.add_argument('--part_column', type=str, default='sl_part_description',
                       help='CSV column for part description used for case-level operating point selection (default: sl_part_description)')
    
    # Regularization and overfitting control options
    parser.add_argument('--latent_dim', type=int, default=256,
                       help='Latent dimension for attention mechanism (smaller = less overfitting)')
    parser.add_argument('--dropout', type=float, default=0.0,
                       help='Dropout rate for regularization (0.0-0.5 recommended)')
    parser.add_argument('--attention_dropout', type=float, default=0.0,
                       help='Dropout rate specifically for attention scores')
    parser.add_argument('--use_layer_norm', action='store_true',
                       help='Use layer normalization for training stability')
    parser.add_argument('--use_batch_norm', action='store_true',
                       help='Use batch normalization (often more effective than layer norm for overfitting)')
    parser.add_argument('--use_skip_connection', action='store_true',
                       help='Use skip connections in attention mechanism')
    parser.add_argument('--early_stopping_patience', type=int, default=None,
                       help='Early stopping patience in epochs (None = disabled)')
    parser.add_argument('--gradient_clip_value', type=float, default=None,
                       help='Gradient clipping value (None = disabled)')
    parser.add_argument('--save_final_model', action='store_true',
                       help='Save final_model.pt at the end of training (default: disabled)')
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


def aggregate_slide_results_to_case_level_for_operating_point(
    slide_identifiers,
    raw_labels,
    raw_probabilities,
    df,
    slide_id_column,
    case_id_column,
    part_column,
    case_level_method,
):
    """
    Aggregate slide-level labels/probabilities to case-part level for operating point selection.
    Returns (y_true_case, y_proba_case).
    """
    raw_labels = np.asarray(raw_labels)
    raw_probabilities = np.asarray(raw_probabilities)
    n_classes = raw_probabilities.shape[1]

    # Build slide -> (case_id, part) map from dataframe
    slide_to_case_part = {}
    for slide_id in slide_identifiers:
        if slide_id in slide_to_case_part:
            continue
        rows = df[df[slide_id_column] == slide_id]
        if len(rows) == 0:
            continue
        row = rows.iloc[0]
        cid = row[case_id_column]
        part = row[part_column]
        slide_to_case_part[slide_id] = (cid, part)

    # Group slide indices by (case_id, part)
    from collections import defaultdict

    groups = defaultdict(list)
    for i, slide_id in enumerate(slide_identifiers):
        key = slide_to_case_part.get(slide_id)
        if key is None:
            continue
        groups[key].append(i)

    case_part_y_true = []
    case_part_y_proba = []

    for (_, _), indices in sorted(groups.items()):
        labels = raw_labels[indices]
        # All slides in same case-part should share the same label
        if len(np.unique(labels)) > 1:
            raise ValueError(
                f"Case-part has slides with different labels: {labels.tolist()}. "
                "Labels must be consistent per case-part."
            )
        case_part_y_true.append(int(labels[0]))

        probs = raw_probabilities[indices]
        if case_level_method == 'mean':
            agg_probs = np.mean(probs, axis=0)
        else:
            agg_probs = np.max(probs, axis=0)
            # Renormalize so probabilities sum to 1
            agg_probs = agg_probs / agg_probs.sum()
        case_part_y_proba.append(agg_probs)

    if not case_part_y_true:
        raise ValueError("No case-part groups could be formed for case-level operating point selection.")

    return np.array(case_part_y_true), np.array(case_part_y_proba)


def create_training_visualizations(training_metrics, output_dir, logger=None):
    """Create visualizations for training metrics."""
    if logger is None:
        logger = get_logger('train_with_balanced_options')
    logger.info("Creating training visualizations...")
    
    # Plot training metrics
    plot_training_metrics(
        training_metrics['train_loss'], training_metrics['val_loss'],
        'Loss', 'Training and Validation Loss',
        os.path.join(output_dir, 'training_loss.png')
    )
    
    plot_training_metrics(
        training_metrics['train_acc'], training_metrics['val_acc'],
        'Accuracy', 'Training and Validation Accuracy',
        os.path.join(output_dir, 'training_accuracy.png')
    )
    
    plot_training_metrics(
        [], training_metrics['val_f1_macro'],
        'F1-Score (Macro)', 'Validation F1-Score (Macro)',
        os.path.join(output_dir, 'validation_f1_macro.png')
    )
    
    plot_training_metrics(
        [], training_metrics['val_roc_auc_macro'],
        'ROC AUC (Macro)', 'Validation ROC AUC (Macro)',
        os.path.join(output_dir, 'validation_roc_auc_macro.png')
    )


def create_evaluation_visualizations(metrics, class_names, output_dir, logger=None):
    """Create visualizations for evaluation metrics."""
    if logger is None:
        logger = get_logger('train_with_balanced_options')
    logger.info("Creating evaluation visualizations...")
    
    # Plot confusion matrix
    plot_confusion_matrix(
        metrics['confusion_matrix'], class_names,
        'Confusion Matrix - Validation Set',
        os.path.join(output_dir, 'confusion_matrix.png')
    )
    
    # Plot per-class metrics
    plot_per_class_metrics(
        metrics['per_class_metrics'].set_index('class'),
        'Per-Class Performance Metrics',
        os.path.join(output_dir, 'per_class_metrics.png')
    )
    
    # For binary classification, create ROC and PRC curves
    is_binary_classification = len(class_names) == 2
    if is_binary_classification:
        logger.info("Creating ROC and Precision-Recall curves for binary classification...")
        # Extract positive class probabilities
        y_true = metrics['raw_labels']
        y_proba = metrics['raw_probabilities'][:, 1]  # Probability of positive class (class 1)
        
        # ROC Curve
        roc_auc = plot_roc_curve(
            y_true, y_proba, class_names,
            f'ROC Curve - {class_names[1]} vs {class_names[0]}',
            os.path.join(output_dir, 'roc_curve.png')
        )
        logger.info(f"ROC curve saved with AUC = {roc_auc:.3f}")
        
        # Precision-Recall Curve
        average_precision = plot_precision_recall_curve(
            y_true, y_proba, class_names,
            f'Precision-Recall Curve - {class_names[1]} vs {class_names[0]}',
            os.path.join(output_dir, 'precision_recall_curve.png')
        )
        logger.info(f"PRC curve saved with AP = {average_precision:.3f}")
        
        # Compare ROC AUC vs AUPRC for imbalanced data insight
        positive_ratio = np.mean(y_true)
        logger.info(f"📊 Binary Classification Curve Comparison:")
        logger.info(f"   Positive class prevalence: {positive_ratio:.1%}")
        logger.info(f"   ROC AUC: {roc_auc:.3f} (focus: TPR vs FPR)")
        logger.info(f"   AUPRC:   {average_precision:.3f} (focus: Precision vs Recall)")
        if positive_ratio < 0.3:  # Imbalanced dataset
            logger.info(f"   💡 For imbalanced data: AUPRC is more informative than ROC AUC")
        if average_precision > roc_auc:
            logger.info(f"   ✅ AUPRC > ROC AUC suggests good precision at high recall")
        elif roc_auc > average_precision + 0.1:
            logger.warning(f"   ROC AUC >> AUPRC may indicate precision challenges")


def print_evaluation_results(metrics, class_names, logger=None):
    """Print comprehensive evaluation results."""
    if logger is None:
        logger = get_logger('train_with_balanced_options')
    is_binary_classification = len(class_names) == 2
    
    log_section(logger, "COMPREHENSIVE EVALUATION RESULTS")
    
    # Print overall metrics
    logger.info("OVERALL PERFORMANCE METRICS:")
    logger.info(f"  Accuracy:                    {metrics['accuracy']:.4f}")
    logger.info(f"  Precision (Macro):           {metrics['precision_macro']:.4f}")
    logger.info(f"  Recall (Macro):              {metrics['recall_macro']:.4f}")
    logger.info(f"  F1-Score (Macro):            {metrics['f1_macro']:.4f}")
    
    if not is_binary_classification:
        logger.info(f"  Precision (Micro):           {metrics['precision_micro']:.4f}")
        logger.info(f"  Recall (Micro):              {metrics['recall_micro']:.4f}")
        logger.info(f"  F1-Score (Micro):            {metrics['f1_micro']:.4f}")
        logger.info(f"  ROC AUC (Macro):             {metrics['roc_auc_macro']:.4f}")
        logger.info(f"  ROC AUC (Micro):             {metrics['roc_auc_micro']:.4f}")
        logger.info(f"  Average Precision (Macro):   {metrics['avg_precision_macro']:.4f}")
        logger.info(f"  Average Precision (Micro):   {metrics['avg_precision_micro']:.4f}")
    else:
        # Binary classification specific metrics
        logger.info(f"  Sensitivity (Recall):        {metrics.get('sensitivity', metrics['recall_macro']):.4f}")
        logger.info(f"  Specificity:                 {metrics.get('specificity', 'N/A')}")
        logger.info(f"  ROC AUC:                     {metrics['roc_auc_macro']:.4f}")
        logger.info(f"  Average Precision:           {metrics['avg_precision_macro']:.4f}")
        logger.info(f"  True Positives:              {metrics.get('true_positives', 'N/A')}")
        logger.info(f"  True Negatives:              {metrics.get('true_negatives', 'N/A')}")
        logger.info(f"  False Positives:             {metrics.get('false_positives', 'N/A')}")
        logger.info(f"  False Negatives:             {metrics.get('false_negatives', 'N/A')}")
    
    logger.info(f"  Matthews Correlation Coeff:  {metrics['mcc']:.4f}")
    
    # Print per-class metrics
    logger.info("PER-CLASS PERFORMANCE:")
    logger.info(f"{'Class':<25} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    logger.info("-" * 60)
    for _, row in metrics['per_class_metrics'].iterrows():
        logger.info(f"{row['class']:<25} {row['precision']:<10.4f} {row['recall']:<10.4f} {row['f1']:<10.4f}")


def save_results(metrics, class_names, val_loader, training_config, balanced_config, output_dir, logger=None):
    """Save detailed results to JSON and CSV files."""
    if logger is None:
        logger = get_logger('train_with_balanced_options')
    logger.info("Saving detailed results...")
    
    is_binary_classification = len(class_names) == 2
    
    # Save comprehensive metrics
    overall_metrics = {
        'accuracy': float(metrics['accuracy']),
        'precision_macro': float(metrics['precision_macro']),
        'recall_macro': float(metrics['recall_macro']),
        'f1_macro': float(metrics['f1_macro']),
        'roc_auc_macro': float(metrics['roc_auc_macro']),
        'avg_precision_macro': float(metrics['avg_precision_macro']),
        'mcc': float(metrics['mcc']),
        'is_binary_classification': is_binary_classification
    }
    
    # Add metrics specific to classification type
    if is_binary_classification:
        overall_metrics.update({
            'sensitivity': float(metrics.get('sensitivity', metrics['recall_macro'])),
            'specificity': float(metrics.get('specificity', 0)),
            'true_positives': int(metrics.get('true_positives', 0)),
            'true_negatives': int(metrics.get('true_negatives', 0)),
            'false_positives': int(metrics.get('false_positives', 0)),
            'false_negatives': int(metrics.get('false_negatives', 0))
        })
    else:
        overall_metrics.update({
            'precision_micro': float(metrics['precision_micro']),
            'recall_micro': float(metrics['recall_micro']),
            'f1_micro': float(metrics['f1_micro']),
            'roc_auc_micro': float(metrics['roc_auc_micro']),
            'avg_precision_micro': float(metrics['avg_precision_micro'])
        })
    
    # Add balanced training configuration to results
    balanced_training_config = {
        'enable_balancing': balanced_config.enable_balancing,
        'balancing_method': balanced_config.balancing_method,
        'sampling_strategy': balanced_config.sampling_strategy,
        'weight_method': balanced_config.weight_method,
        'pos_weight': balanced_config.pos_weight,
        'smooth_weights': balanced_config.smooth_weights
    }
    
    results = {
        'overall_metrics': overall_metrics,
        'per_class_metrics': {
            row['class']: {
                'precision': float(row['precision']),
                'recall': float(row['recall']),
                'f1': float(row['f1'])
            } for _, row in metrics['per_class_metrics'].iterrows()
        },
        'confusion_matrix': metrics['confusion_matrix'].tolist(),
        'class_names': to_json_serializable(list(class_names)),
        'total_validation_samples': len(val_loader.dataset),
        'training_config': training_config,
        'balanced_training_config': balanced_training_config
    }
    
    with open(os.path.join(output_dir, 'comprehensive_evaluation.json'), 'w') as f:
        json.dump(to_json_serializable(results), f, indent=2)
    
    # Save per-class metrics as CSV
    metrics['per_class_metrics'].to_csv(
        os.path.join(output_dir, 'per_class_metrics.csv'), index=False
    )


def analyze_dataset(master_df):
    """Analyze and print dataset statistics."""
    print(f"\nDataset Analysis:")
    print(f"Total slides: {len(master_df)}")
    print(f"Unique patients (MRN): {master_df['MRN'].nunique()}")
    print(f"Unique diagnoses: {master_df['diagnosis'].nunique()}")
    
    # Show class distribution
    print(f"\nClass Distribution:")
    class_counts = master_df['diagnosis'].value_counts().sort_index()
    for diagnosis, count in class_counts.items():
        print(f"  {diagnosis}: {count} slides")
    
    # Check for patients with multiple slides
    patient_slide_counts = master_df.groupby('MRN').size()
    patients_with_multiple_slides = (patient_slide_counts > 1).sum()
    max_slides_per_patient = patient_slide_counts.max()
    
    print(f"\nPatient-Slide Relationship:")
    print(f"Patients with multiple slides: {patients_with_multiple_slides}")
    print(f"Maximum slides per patient: {max_slides_per_patient}")
    
    if patients_with_multiple_slides > 0:
        print("✓ Patient-level splitting is crucial to prevent data leakage!")
    else:
        print("→ Patient-level splitting provides the same result as slide-level splitting.")
    print()


def main():
    """Main training function with balanced training options."""
    args = parse_arguments()
    
    # Set up logging
    logger = setup_logger('train_with_balanced_options', args.output_dir)
    
    # Set seed for reproducibility
    set_seed(args.random_state)
    
    log_section(logger, "TRAINING WITH CONFIGURABLE CLASS BALANCING")
    logger.info("Enhanced training script with natural sampling + class-weighted loss options")
    
    # Log arguments
    log_args(logger, args, "Balanced Training")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load master dataframe
    master_df = pd.read_csv(args.master_df_path)
    # By default, use the full master_df for training/validation unless a held-out test split is requested
    trainval_df = master_df
    model_save_dir = os.path.join(args.output_dir, 'models')
    
    # Set training parameters
    training_config = {
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'val_split': args.val_split,
        'test_split': args.test_split,
        'random_state': args.random_state,
        'splitting_method': 'patient_level'
    }
    
    # Configure balanced training
    balanced_config = BalancedTrainingConfig(
        enable_balancing=args.enable_balancing,
        balancing_method=args.balancing_method,
        sampling_strategy=args.sampling_strategy,
        weight_method=args.weight_method,
        pos_weight=args.pos_weight,
        smooth_weights=args.smooth_weights
    )
    
    # Configure metric selection
    metric_mode = infer_metric_mode(args.best_metric) if args.metric_mode == 'auto' else args.metric_mode
    logger.info("Best Model Selection Configuration:")
    logger.info(f"  Metric: {args.best_metric}")
    logger.info(f"  Mode: {metric_mode}imize")
    if args.best_metric == 'val_auprc_macro':
        logger.info(f"  📊 AUPRC is especially good for imbalanced datasets!")
    
    # Configure regularization
    logger.info("Regularization Configuration:")
    logger.info(f"  Latent dimension: {args.latent_dim}")
    logger.info(f"  Dropout: {args.dropout}")
    logger.info(f"  Attention dropout: {args.attention_dropout}")
    logger.info(f"  Layer normalization: {args.use_layer_norm}")
    logger.info(f"  Batch normalization: {args.use_batch_norm}")
    logger.info(f"  Skip connections: {args.use_skip_connection}")
    
    # Normalization recommendations
    if args.use_batch_norm and args.use_layer_norm:
        logger.warning(f"Both normalizations enabled - LayerNorm takes precedence")
    elif not args.use_batch_norm and not args.use_layer_norm:
        logger.info(f"💡 Consider batch norm (--use_batch_norm) for stronger overfitting control")
    logger.info(f"  Early stopping: {'Enabled (' + str(args.early_stopping_patience) + ' epochs)' if args.early_stopping_patience else 'Disabled'}")
    logger.info(f"  Gradient clipping: {'Enabled (' + str(args.gradient_clip_value) + ')' if args.gradient_clip_value else 'Disabled'}")
    
    # Provide recommendations based on regularization settings
    if args.dropout == 0.0 and args.attention_dropout == 0.0:
        logger.info(f"💡 Consider adding dropout (--dropout 0.2) to reduce overfitting")
    if args.early_stopping_patience is None:
        logger.info(f"💡 Consider early stopping (--early_stopping_patience 5) to prevent overfitting")
    if args.latent_dim >= 256:
        logger.info(f"💡 Consider smaller latent_dim (--latent_dim 64) to reduce model complexity")
    
    # If a held-out patient-level test split is requested, carve it out and save to CSV
    if args.test_split and args.test_split > 0.0:
        from modules.feature_dataset import patient_level_split

        logger.info(f"Creating patient-level test split with test_split={args.test_split}, "
                    f"val_split={args.val_split}")
        train_slides, val_slides, test_slides = patient_level_split(
            master_df,
            val_split=args.val_split,
            test_split=args.test_split,
            random_state=args.random_state,
            stratify_by_slide_distribution=True,
            max_iterations=100,
        )

        # Build test dataframe and save it in a format mirroring the input master CSV
        test_df = master_df[master_df['slide_name'].isin(test_slides)].copy()
        test_csv_path = os.path.join(args.output_dir, 'test_split_patient_level.csv')
        logger.info(f"Saving held-out test set with {len(test_df)} slides to: {test_csv_path}")
        test_df.to_csv(test_csv_path, index=False)

        # Exclude test slides from the data used for training/validation
        trainval_df = master_df[~master_df['slide_name'].isin(test_slides)].copy()
        logger.info(f"Training/validation will use {len(trainval_df)} slides "
                    f"(excluding {len(test_df)} held-out test slides).")

    # Analyze dataset (training/validation subset if test split was created)
    analyze_dataset(trainval_df)
    
    # Get class information from the dataset
    unique_diagnoses = sorted(trainval_df['diagnosis'].unique())
    class_names = unique_diagnoses
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Optional fast path: recompute operating point on existing best_model without retraining
    if args.recompute_operating_point:
        logger.info("Recomputing operating point on validation set without retraining.")
        # Recreate the same patient-level validation split used during training:
        # - Same trainval_df (master_df minus held-out test, if any)
        # - Same val_split and random_state
        logger.info("Creating validation dataloader for operating point recomputation...")
        _, val_loader, _ = create_patient_level_dataloaders(
            h5_directory=args.h5_directory,
            master_df=trainval_df,
            batch_size=training_config['batch_size'],
            num_workers=4,
            val_split=training_config['val_split'],
            test_split=0.0,
            random_state=training_config['random_state'],
            stratify_by_slide_distribution=True,
            max_iterations=100
        )

        if val_loader is None:
            logger.warning("Validation dataloader is None; cannot recompute operating point.")
            return

        # Load the previously trained best model
        checkpoint_path = os.path.join(model_save_dir, 'best_model.pt')
        if not os.path.exists(checkpoint_path):
            logger.warning(f"No best_model.pt found at {checkpoint_path}; cannot recompute operating point.")
            return

        checkpoint = torch.load(checkpoint_path, weights_only=False)
        model_config = checkpoint.get('model_config', {})
        # Use class_names from checkpoint if present to be fully consistent
        ckpt_class_names = checkpoint.get('class_names', class_names)
        if ckpt_class_names is not None:
            class_names = ckpt_class_names

        best_model = GatedMILClassifier(
            d_model=1024,
            num_classes=len(class_names),
            latent_dim=model_config.get('latent_dim', args.latent_dim),
            dropout=model_config.get('dropout', args.dropout),
            attention_dropout=model_config.get('attention_dropout', args.attention_dropout),
            use_layer_norm=model_config.get('use_layer_norm', args.use_layer_norm),
            use_batch_norm=model_config.get('use_batch_norm', args.use_batch_norm),
            use_skip_connection=model_config.get('use_skip_connection', args.use_skip_connection)
        )
        best_model.load_state_dict(checkpoint['model_state_dict'])
        best_model = best_model.to(device)

        # Reuse the same operating point selection logic (now with optional case-level mode)
        if args.operating_point_selection != 'default' and len(class_names) == 2:
            logger.info(f"Finding optimal threshold using method: {args.operating_point_selection}")
            try:
                # Optional case-level aggregation for operating point selection
                if args.case_level:
                    # Validate required columns exist in training dataframe
                    for col, name in [
                        ('slide_name', 'slide_name (fixed column used for H5/slide mapping)'),
                        (args.case_id_column, 'case_id_column'),
                        (args.part_column, 'part_column'),
                    ]:
                        if col not in trainval_df.columns:
                            raise ValueError(
                                f"Case-level operating point selection requires column '{col}' ({name}) in training CSV. "
                                f"Available columns: {list(trainval_df.columns)}"
                            )

                    # Recover validation slide identifiers in dataloader order
                    val_dataset = val_loader.dataset
                    if not hasattr(val_dataset, 'h5_files'):
                        raise ValueError(
                            "Validation dataset does not expose 'h5_files'; cannot recover slide identifiers "
                            "for case-level operating point selection."
                        )
                    slide_identifiers = [
                        os.path.basename(f).replace('.h5', '.svs') for f in val_dataset.h5_files
                    ]

                    # Collect slide-level labels and probabilities in the same order
                    all_labels = []
                    all_probs = []
                    with torch.no_grad():
                        for batch_idx, (features, labels) in enumerate(val_loader):
                            features = features.to(device)
                            labels = labels.to(device).long()

                            outputs = best_model(features)
                            probs = torch.softmax(outputs, dim=1).cpu().numpy()

                            all_probs.extend(probs)
                            all_labels.extend(labels.cpu().numpy())

                    all_probs = np.array(all_probs)
                    all_labels = np.array(all_labels)

                    if all_probs.shape[0] != len(slide_identifiers):
                        raise ValueError(
                            f"Mismatch between number of validation predictions ({all_probs.shape[0]}) "
                            f"and slide identifiers ({len(slide_identifiers)})."
                        )

                    # Aggregate to case-part level
                    y_true_case, y_proba_case = aggregate_slide_results_to_case_level_for_operating_point(
                        slide_identifiers=slide_identifiers,
                        raw_labels=all_labels,
                        raw_probabilities=all_probs,
                        df=trainval_df,
                        slide_id_column='slide_name',
                        case_id_column=args.case_id_column,
                        part_column=args.part_column,
                        case_level_method=args.case_level_method,
                    )

                    if y_proba_case.shape[1] != 2:
                        raise ValueError(
                            "Case-level operating point selection currently only supports binary classification."
                        )

                    y_true = y_true_case
                    y_proba_pos = y_proba_case[:, 1]
                else:
                    # Slide-level threshold optimization (reuse utility)
                    optimal_threshold = find_optimal_threshold(
                        best_model, val_loader, class_names, device,
                        method=args.operating_point_selection
                    )
                    # Write and return
                    checkpoint['optimal_threshold'] = float(optimal_threshold)
                    checkpoint['operating_point_method'] = args.operating_point_selection
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"✓ Optimal threshold (slide-level) updated to: {optimal_threshold:.4f}")
                    return

                # Case-level threshold optimization on y_true/y_proba_pos
                if args.operating_point_selection == 'youden':
                    fpr, tpr, thresholds = roc_curve(y_true, y_proba_pos)
                    youden_j = tpr - fpr
                    optimal_idx = np.argmax(youden_j)
                    optimal_threshold = thresholds[optimal_idx]
                elif args.operating_point_selection == 'f1':
                    precision, recall, thresholds = precision_recall_curve(y_true, y_proba_pos)
                    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
                    optimal_idx = np.argmax(f1_scores)
                    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
                elif args.operating_point_selection == 'sensitivity_95':
                    fpr, tpr, thresholds = roc_curve(y_true, y_proba_pos)
                    target_sensitivity = 0.95
                    optimal_idx = np.argmin(np.abs(tpr - target_sensitivity))
                    optimal_threshold = thresholds[optimal_idx]
                else:
                    raise ValueError(
                        f"Unknown operating point selection method: {args.operating_point_selection}. "
                        "Choose from: 'youden', 'f1', 'sensitivity_95'."
                    )

                checkpoint['optimal_threshold'] = float(optimal_threshold)
                checkpoint['operating_point_method'] = args.operating_point_selection
                torch.save(checkpoint, checkpoint_path)
                mode_str = "case-level" if args.case_level else "slide-level"
                logger.info(
                    f"✓ Optimal threshold ({mode_str}, method={args.operating_point_selection}) "
                    f"updated to: {optimal_threshold:.4f}"
                )
            except Exception as e:
                logger.warning(f"Failed to recompute optimal threshold: {e}")
                logger.warning("Continuing without updating threshold...")
        else:
            logger.info("Operating point selection is 'default' or not binary classification; no threshold recomputed.")

        return

    if args.eval_only:
        logger.info("Skipping training; running validation evaluation on existing best_model.pt")
        checkpoint_path = os.path.join(model_save_dir, 'best_model.pt')
        if not os.path.exists(checkpoint_path):
            logger.error(f"No best_model.pt found at {checkpoint_path}; cannot run eval_only.")
            return
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        ckpt_class_names = checkpoint.get('class_names', class_names)
        if ckpt_class_names is not None:
            class_names = ckpt_class_names
    else:
        # Train the model with balanced training configuration
        logger.info("Starting training with balanced configuration...")
        model, training_metrics, class_names = train_model(
            h5_directory=args.h5_directory,
            master_df=trainval_df,
            num_epochs=training_config['num_epochs'],
            batch_size=training_config['batch_size'],
            learning_rate=training_config['learning_rate'],
            weight_decay=training_config['weight_decay'],
            val_split=training_config['val_split'],
            device=device,
            model_save_dir=model_save_dir,
            use_patient_level_split=True,
            random_state=training_config['random_state'],
            balanced_config=balanced_config,
            best_metric=args.best_metric,
            metric_mode=metric_mode,
            latent_dim=args.latent_dim,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            use_layer_norm=args.use_layer_norm,
            use_batch_norm=args.use_batch_norm,
            use_skip_connection=args.use_skip_connection,
            early_stopping_patience=args.early_stopping_patience,
            gradient_clip_value=args.gradient_clip_value,
            save_final_model=args.save_final_model,
            save_best_model=args.save_best_model,
            logger=logger if args.save_training_logs_too else None,
            save_training_logs_too=args.save_training_logs_too
        )

        # Create training visualizations
        create_training_visualizations(training_metrics, args.output_dir, logger)
    
    # Create validation dataloader for final evaluation
    logger.info("Creating validation dataloader for final evaluation...")
    _, val_loader, _ = create_patient_level_dataloaders(
        h5_directory=args.h5_directory,
        master_df=trainval_df,
        batch_size=training_config['batch_size'],
        num_workers=4,
        val_split=training_config['val_split'],
        test_split=0.0,
        random_state=training_config['random_state'],
        stratify_by_slide_distribution=True,
        max_iterations=100
    )
    
    # Load the best model for evaluation
    checkpoint = torch.load(os.path.join(model_save_dir, 'best_model.pt'), weights_only=False)
    model_config = checkpoint.get('model_config', {})
    
    best_model = GatedMILClassifier(
        d_model=1024, 
        num_classes=len(class_names), 
        latent_dim=model_config.get('latent_dim', args.latent_dim),
        dropout=model_config.get('dropout', args.dropout),
        attention_dropout=model_config.get('attention_dropout', args.attention_dropout),
        use_layer_norm=model_config.get('use_layer_norm', args.use_layer_norm),
        use_batch_norm=model_config.get('use_batch_norm', args.use_batch_norm),
        use_skip_connection=model_config.get('use_skip_connection', args.use_skip_connection)
    )
    best_model.load_state_dict(checkpoint['model_state_dict'])
    best_model = best_model.to(device)
    
    # Comprehensive evaluation
    metrics = evaluate_model(best_model, val_loader, class_names, device)
    
    # Find optimal threshold if requested and binary classification
    if args.operating_point_selection != 'default' and len(class_names) == 2 and val_loader is not None:
        logger.info(f"\nFinding optimal threshold using method: {args.operating_point_selection}")
        try:
            # Optional case-level aggregation for operating point selection
            if args.case_level:
                # Validate required columns exist in training dataframe
                for col, name in [
                    ('slide_name', 'slide_name (fixed column used for H5/slide mapping)'),
                    (args.case_id_column, 'case_id_column'),
                    (args.part_column, 'part_column'),
                ]:
                    if col not in trainval_df.columns:
                        raise ValueError(
                            f"Case-level operating point selection requires column '{col}' ({name}) in training CSV. "
                            f"Available columns: {list(trainval_df.columns)}"
                        )

                # Recover validation slide identifiers in dataloader order
                val_dataset = val_loader.dataset
                if not hasattr(val_dataset, "h5_files"):
                    raise ValueError(
                        "Validation dataset does not expose 'h5_files'; cannot recover slide identifiers "
                        "for case-level operating point selection."
                    )
                slide_identifiers = [
                    os.path.basename(f).replace(".h5", ".svs") for f in val_dataset.h5_files
                ]

                # Collect slide-level labels and probabilities in the same order
                all_labels = []
                all_probs = []
                with torch.no_grad():
                    for batch_idx, (features, labels) in enumerate(val_loader):
                        features = features.to(device)
                        labels = labels.to(device).long()

                        outputs = best_model(features)
                        probs = torch.softmax(outputs, dim=1).cpu().numpy()

                        all_probs.extend(probs)
                        all_labels.extend(labels.cpu().numpy())

                all_probs = np.array(all_probs)
                all_labels = np.array(all_labels)

                if all_probs.shape[0] != len(slide_identifiers):
                    raise ValueError(
                        f"Mismatch between number of validation predictions ({all_probs.shape[0]}) "
                        f"and slide identifiers ({len(slide_identifiers)})."
                    )

                # Aggregate to case-part level
                y_true_case, y_proba_case = aggregate_slide_results_to_case_level_for_operating_point(
                    slide_identifiers=slide_identifiers,
                    raw_labels=all_labels,
                    raw_probabilities=all_probs,
                    df=trainval_df,
                    slide_id_column='slide_name',
                    case_id_column=args.case_id_column,
                    part_column=args.part_column,
                    case_level_method=args.case_level_method,
                )

                if y_proba_case.shape[1] != 2:
                    raise ValueError(
                        "Case-level operating point selection currently only supports binary classification."
                    )

                y_true = y_true_case
                y_proba_pos = y_proba_case[:, 1]

                # Apply the same threshold optimization logic but on case-level arrays
                if args.operating_point_selection == "youden":
                    fpr, tpr, thresholds = roc_curve(y_true, y_proba_pos)
                    youden_j = tpr - fpr
                    optimal_idx = np.argmax(youden_j)
                    optimal_threshold = thresholds[optimal_idx]
                elif args.operating_point_selection == "f1":
                    precision, recall, thresholds = precision_recall_curve(y_true, y_proba_pos)
                    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
                    optimal_idx = np.argmax(f1_scores)
                    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
                elif args.operating_point_selection == "sensitivity_95":
                    fpr, tpr, thresholds = roc_curve(y_true, y_proba_pos)
                    target_sensitivity = 0.95
                    optimal_idx = np.argmin(np.abs(tpr - target_sensitivity))
                    optimal_threshold = thresholds[optimal_idx]
                else:
                    raise ValueError(
                        f"Unknown operating point selection method: {args.operating_point_selection}. "
                        "Choose from: 'youden', 'f1', 'sensitivity_95'."
                    )
                logger.info(
                    f"✓ Optimal threshold (case-level, method={args.operating_point_selection}, "
                    f"agg={args.case_level_method}) found: {optimal_threshold:.4f}"
                )
            else:
                # Default slide-level operating point selection
                optimal_threshold = find_optimal_threshold(
                    best_model, val_loader, class_names, device,
                    method=args.operating_point_selection
                )
                logger.info(f"✓ Optimal threshold found: {optimal_threshold:.4f}")

            # Update checkpoint with optimal threshold
            checkpoint_path = os.path.join(model_save_dir, 'best_model.pt')
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            checkpoint['optimal_threshold'] = float(optimal_threshold)
            checkpoint['operating_point_method'] = args.operating_point_selection
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"✓ Checkpoint updated with optimal threshold: {optimal_threshold:.4f}")
        except Exception as e:
            logger.warning(f"Failed to find optimal threshold: {e}")
            logger.warning("Continuing without threshold optimization...")
    elif args.operating_point_selection != 'default' and len(class_names) != 2:
        logger.info(f"Threshold optimization skipped (only supported for binary classification)")
    
    # Print evaluation results
    print_evaluation_results(metrics, class_names, logger)
    
    # Create evaluation visualizations
    create_evaluation_visualizations(metrics, class_names, args.output_dir, logger)
    
    # Save detailed results
    save_results(metrics, class_names, val_loader, training_config, balanced_config, args.output_dir)
    
    # Final summary
    log_section(logger, "TRAINING COMPLETED SUCCESSFULLY!")
    classification_type = "Binary" if len(class_names) == 2 else "Multi-class"
    logger.info(f"✓ {classification_type} model trained on {len(class_names)} classes")
    logger.info(f"✓ Patient-level splitting prevents data leakage")
    logger.info(f"✓ Balanced training: {'Enabled' if balanced_config.enable_balancing else 'Disabled'}")
    if balanced_config.enable_balancing:
        logger.info(f"  - Method: {balanced_config.balancing_method}")
        logger.info(f"  - Sampling: {balanced_config.sampling_strategy}")
        logger.info(f"  - Weight method: {balanced_config.weight_method}")
    logger.info(f"✓ Comprehensive evaluation metrics calculated")
    logger.info(f"✓ Results saved to: {args.output_dir}")
    logger.info(f"✓ Best model accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"✓ Best model F1-Score (Macro): {metrics['f1_macro']:.4f}")
    if len(class_names) == 2:
        logger.info(f"✓ ROC AUC: {metrics['roc_auc_macro']:.4f}")
        logger.info(f"✓ AUPRC: {metrics['avg_precision_macro']:.4f}")
        logger.info(f"✓ ROC curve saved to: {os.path.join(args.output_dir, 'roc_curve.png')}")
        logger.info(f"✓ PRC curve saved to: {os.path.join(args.output_dir, 'precision_recall_curve.png')}")
        if metrics['avg_precision_macro'] > metrics['roc_auc_macro']:
            logger.info(f"✅ AUPRC > ROC AUC indicates good precision-recall performance")
        logger.info(f"📊 For imbalanced data, focus on AUPRC over ROC AUC")


if __name__ == "__main__":
    main()
