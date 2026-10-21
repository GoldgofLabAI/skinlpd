"""
Test evaluation script for trained MIL models.

This script loads a trained model and evaluates it on a test set provided via CSV file.
It generates comprehensive metrics, visualizations, and saves detailed results.
"""
import torch
import pandas as pd
import numpy as np
import os
import sys
import json
import argparse
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.utils import resample

# Add this package directory to Python path so `from modules import ...` works.
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import modules
from modules import (
    # Core functionality
    GatedMILClassifier, FeatureDataset, create_test_only_dataloader,
    # Evaluation
    evaluate_model,
    # Metrics and visualization
    plot_confusion_matrix, plot_per_class_metrics, 
    plot_roc_curve, plot_precision_recall_curve,
    calculate_comprehensive_metrics,
    # Utilities
    set_seed, to_json_serializable
)

import warnings
warnings.filterwarnings('ignore')


def parse_arguments():
    """Parse command line arguments for test evaluation."""
    parser = argparse.ArgumentParser(description='Evaluate trained model on test set')
    
    # Required parameters
    parser.add_argument('--test_csv_path', type=str, required=True,
                       help='Path to CSV file containing test set data')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint (.pt file)')
    parser.add_argument('--h5_directory', type=str, required=True,
                       help='Directory containing H5 feature files')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for evaluation results')
    
    # Optional parameters
    parser.add_argument('--batch_size', type=int, default=16, 
                       help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of worker processes for data loading')
    parser.add_argument('--device', type=str, default='cuda:3',
                       choices=['auto', 'cpu', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'],
                       help='Device to use for evaluation (auto=auto-detect)')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random seed for reproducibility')
    
    # Analysis options
    parser.add_argument('--analyze_class_distribution', action='store_true',
                       help='Analyze and print class distribution in test set')
    parser.add_argument('--save_predictions', action='store_true',
                       help='Save individual predictions and probabilities to CSV')
    parser.add_argument('--create_detailed_plots', action='store_true',
                       help='Create additional detailed analysis plots')
    parser.add_argument('--slide_id_column', type=str, default='slide_name',
                       help='Column name in CSV to use as slide identifier (default: slide_name)')
    
    # Case-level evaluation options
    parser.add_argument('--case_level', action='store_true',
                       help='Evaluate at case-part level: aggregate slide probs by (case_id, part) then evaluate')
    parser.add_argument('--case_level_method', type=str, default='mean',
                       choices=['mean', 'max'],
                       help='How to aggregate slide probabilities per case-part (default: mean)')
    parser.add_argument('--case_id_column', type=str, default='sp_specimen_id',
                       help='CSV column for case/specimen ID (default: sp_specimen_id)')
    parser.add_argument('--part_column', type=str, default='sl_part_description',
                       help='CSV column for part description (default: sl_part_description)')
    
    # Bootstrapping options
    parser.add_argument('--perform_bootstrapping', action='store_true',
                       help='Perform bootstrap resampling to calculate confidence intervals for metrics')
    parser.add_argument('--bootstrap_iterations', type=int, default=1000,
                       help='Number of bootstrap iterations (default: 1000)')
    
    # Visualization options
    parser.add_argument('--fontsize', type=int, default=20,
                       help='Font size for plot text (default: 16)')
    parser.add_argument('--color_theme', type=str, default='red', 
                       choices=['red', 'blue', 'green', 'purple', 'orange'],
                       help='Color theme for plots (default: red)')
    parser.add_argument('--font_family', type=str, default='DejaVu Sans',
                       help='Font family for plots (default: DejaVu Sans)')
    
    # Checkpoint compatibility options
    parser.add_argument('--strict_checkpoint', action='store_true',
                       help='Require class names in checkpoint; otherwise fail')
    parser.add_argument('--override_class_names', type=str, default=None,
                       help='Comma-separated class names to use if missing in checkpoint (e.g., "negative,positive")')
    
    return parser.parse_args()


def load_trained_model(model_path, device, override_class_names=None, strict_checkpoint=False):
    """
    Load a trained model from checkpoint.
    
    Args:
        model_path (str): Path to model checkpoint
        device (torch.device): Device to load model on
        override_class_names (str, optional): Comma-separated class names to override
        strict_checkpoint (bool): Whether to require class names in checkpoint
    
    Returns:
        tuple: (model, model_config, class_names, optimal_threshold)
    """
    print(f"Loading trained model from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Extract model configuration
    model_config = checkpoint.get('model_config', {})
    class_names = checkpoint.get('class_names', None)
    
    # Handle missing class names gracefully
    if class_names is None:
        if override_class_names:
            class_names = [c.strip() for c in override_class_names.split(',') if c.strip()]
            print(f"⚠️  Class names missing in checkpoint. Using override class names: {class_names}")
        elif strict_checkpoint:
            raise ValueError("Class names not found in checkpoint and strict mode enabled. Provide --override_class_names or retrain with class info.")
        else:
            # Default to binary classification with conventional ordering
            class_names = ['negative', 'positive']
            print("⚠️  Class names not found in checkpoint. Defaulting to binary classes: ['negative', 'positive']")
    
    # Get model parameters
    d_model = model_config.get('d_model', 1024)
    num_classes = len(class_names)
    latent_dim = model_config.get('latent_dim', 256)
    dropout = model_config.get('dropout', 0.0)
    attention_dropout = model_config.get('attention_dropout', 0.0)
    use_layer_norm = model_config.get('use_layer_norm', False)
    use_batch_norm = model_config.get('use_batch_norm', False)
    use_skip_connection = model_config.get('use_skip_connection', False)
    
    print(f"Model configuration:")
    print(f"  - Input dimension: {d_model}")
    print(f"  - Number of classes: {num_classes}")
    print(f"  - Class names: {class_names}")
    print(f"  - Latent dimension: {latent_dim}")
    print(f"  - Dropout: {dropout}")
    print(f"  - Attention dropout: {attention_dropout}")
    print(f"  - Layer normalization: {use_layer_norm}")
    print(f"  - Batch normalization: {use_batch_norm}")
    print(f"  - Skip connections: {use_skip_connection}")
    
    # Create model instance
    model = GatedMILClassifier(
        d_model=d_model,
        num_classes=num_classes,
        latent_dim=latent_dim,
        dropout=dropout,
        attention_dropout=attention_dropout,
        use_layer_norm=use_layer_norm,
        use_batch_norm=use_batch_norm,
        use_skip_connection=use_skip_connection
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Check for optimal threshold in checkpoint
    optimal_threshold = checkpoint.get('optimal_threshold', None)
    operating_point_method = checkpoint.get('operating_point_method', None)
    
    if optimal_threshold is not None:
        method_str = f" (method: {operating_point_method})" if operating_point_method else ""
        print(f"✓ Using optimal threshold: {optimal_threshold:.4f}{method_str}")
    else:
        print("✓ No optimal threshold found, using default 0.5 (argmax)")
    
    print("✓ Model loaded successfully")
    return model, model_config, class_names, optimal_threshold, operating_point_method


def create_test_dataloader(test_csv_path, h5_directory, batch_size, num_workers, random_state, slide_id_column='slide_name', class_names=None):
    """
    Create a dataloader for the test set.
    
    Args:
        test_csv_path (str): Path to test CSV file
        h5_directory (str): Directory containing H5 feature files
        batch_size (int): Batch size
        num_workers (int): Number of workers
        random_state (int): Random seed
        slide_id_column (str): Column name to use as slide identifier
    
    Returns:
        tuple: (test_dataloader, test_df, slide_identifiers)
    """
    print(f"Creating test dataloader from: {test_csv_path}")
    
    # Load test dataframe
    test_df = pd.read_csv(test_csv_path)
    print(f"Test set loaded: {len(test_df)} samples")
    print(f"Unique patients: {test_df['MRN'].nunique()}")
    print(f"Unique diagnoses: {test_df['diagnosis'].nunique()}")
    
    # Validate slide_id_column exists
    if slide_id_column not in test_df.columns:
        raise ValueError(f"Column '{slide_id_column}' not found in test CSV. Available columns: {list(test_df.columns)}")
    
    print(f"Using '{slide_id_column}' as slide identifier")
    
    # Check for missing H5 files before creating dataloader
    import glob
    all_h5_files = glob.glob(os.path.join(h5_directory, '*.h5'))
    h5_slide_names = {os.path.basename(f).replace('.h5', '.svs') for f in all_h5_files}
    csv_slide_names = set(test_df[slide_id_column].unique())
    csv_unique_slides = len(csv_slide_names)
    
    # Check for duplicates in CSV
    if len(test_df) != csv_unique_slides:
        duplicate_count = len(test_df) - csv_unique_slides
        print(f"\n📊 CSV Analysis:")
        print(f"   Total rows in CSV: {len(test_df)}")
        print(f"   Unique slides in CSV: {csv_unique_slides}")
        print(f"   Duplicate rows: {duplicate_count}")
    
    missing_slides = csv_slide_names - h5_slide_names
    if missing_slides:
        print(f"\n⚠️  WARNING: {len(missing_slides)} slides in CSV don't have corresponding H5 files:")
        expected_samples = csv_unique_slides - len(missing_slides)
        print(f"   CSV has {csv_unique_slides} unique slides, but only {expected_samples} have H5 files")
        print(f"   This explains the drop from {csv_unique_slides} to {expected_samples} samples")
        
        # Show sample of missing slides
        missing_list = sorted(list(missing_slides))[:10]
        for slide in missing_list:
            slide_rows = test_df[test_df[slide_id_column] == slide]
            diagnosis = slide_rows['diagnosis'].iloc[0] if len(slide_rows) > 0 else 'unknown'
            mrn = slide_rows['MRN'].iloc[0] if 'MRN' in slide_rows.columns and len(slide_rows) > 0 else 'unknown'
            print(f"   - {slide} (diagnosis: {diagnosis}, MRN: {mrn})")
        if len(missing_slides) > 10:
            print(f"   ... and {len(missing_slides) - 10} more")
        
        # Check if there are H5 files that don't match CSV (for debugging)
        extra_h5_slides = h5_slide_names - csv_slide_names
        if extra_h5_slides:
            print(f"\n   Note: There are {len(extra_h5_slides)} H5 files that don't match any CSV entries")
    
    # Create test dataloader using simplified function.
    # Pass class_names (from the trained checkpoint) so the diagnosis->index mapping
    # matches the model's training order; essential for multiclass tasks where the
    # test set may not contain every training class.
    test_dataloader, test_df, slide_identifiers = create_test_only_dataloader(
        h5_directory=h5_directory,
        test_df=test_df,
        batch_size=batch_size,
        num_workers=num_workers,
        class_names=class_names
    )
    
    if test_dataloader is None:
        raise ValueError("Failed to create test dataloader. Check your test CSV and H5 directory.")
    
    print(f"✓ Test dataloader created: {len(test_dataloader.dataset)} samples")
    return test_dataloader, test_df, slide_identifiers


def evaluate_model_with_identifiers(model, dataloader, class_names, slide_identifiers, device=None, optimal_threshold=None):
    """
    Evaluate the trained model on a dataset with comprehensive metrics and slide identifiers.
    
    Args:
        model (nn.Module): Trained model
        dataloader (DataLoader): DataLoader for test dataset
        class_names (list): List of class names
        slide_identifiers (list): List of slide identifiers in the same order as dataloader
        device (torch.device): Device to use for evaluation
        optimal_threshold (float, optional): Optimal threshold for binary classification. If None, uses argmax.
    
    Returns:
        dict: Dictionary of comprehensive evaluation metrics with slide identifiers
    """
    if device is None:
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    all_slide_ids = []
    
    with torch.no_grad():
        for batch_idx, (features, labels) in enumerate(dataloader):
            features = features.to(device)
            labels = labels.to(device).long()
            
            # Forward pass
            outputs = model(features)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            
            # Use threshold-based prediction for binary classification if threshold is available
            if optimal_threshold is not None and len(class_names) == 2:
                # Binary classification with custom threshold
                preds = (probs[:, 1] >= optimal_threshold).astype(int)
            else:
                # Default behavior: argmax (works for multi-class and default threshold)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
            # Add slide identifiers for this batch
            batch_start_idx = batch_idx * dataloader.batch_size
            batch_end_idx = min(batch_start_idx + dataloader.batch_size, len(slide_identifiers))
            all_slide_ids.extend(slide_identifiers[batch_start_idx:batch_end_idx])
    
    # Calculate comprehensive metrics
    all_probs = np.array(all_probs)
    metrics = calculate_comprehensive_metrics(all_labels, all_preds, all_probs, class_names)
    
    # Add raw predictions, probabilities, and slide identifiers
    metrics['raw_labels'] = all_labels
    metrics['raw_predictions'] = all_preds
    metrics['raw_probabilities'] = all_probs
    metrics['slide_identifiers'] = all_slide_ids
    
    return metrics


def aggregate_slide_results_to_case_level(slide_identifiers, raw_labels, raw_probabilities, test_df,
                                          slide_id_column, case_id_column, part_column, class_names,
                                          case_level_method, optimal_threshold=None):
    """
    Aggregate slide-level predictions to case-part level by grouping slides by (case_id, part)
    and aggregating probabilities with mean or max.
    
    Args:
        slide_identifiers: List of slide IDs in same order as raw_labels/raw_probabilities
        raw_labels: Slide-level true labels (int indices)
        raw_probabilities: Slide-level probability arrays, shape (n_slides, n_classes)
        test_df: Full test CSV with slide_id_column, case_id_column, part_column, and diagnosis
        slide_id_column: Column name for slide ID in test_df
        case_id_column: Column name for case/specimen ID
        part_column: Column name for part description
        class_names: List of class names (for label mapping)
        case_level_method: 'mean' or 'max' for aggregating probabilities
        optimal_threshold: For binary classification, threshold for positive class (optional)
    
    Returns:
        tuple: (metrics_dict, case_part_identifiers, case_part_metadata)
        metrics_dict has the same structure as from calculate_comprehensive_metrics plus
        raw_labels, raw_predictions, raw_probabilities, slide_identifiers (replaced by case_part_identifiers).
        case_part_metadata: list of dicts with case_id, part_description, n_slides (and optionally slide_ids).
    """
    raw_labels = np.asarray(raw_labels)
    raw_probabilities = np.asarray(raw_probabilities)
    n_classes = raw_probabilities.shape[1]
    
    # Build slide -> (case_id, part) from test_df; keep first row per slide
    slide_to_case_part = {}
    for slide_id in slide_identifiers:
        if slide_id in slide_to_case_part:
            continue
        rows = test_df[test_df[slide_id_column] == slide_id]
        if len(rows) == 0:
            continue
        row = rows.iloc[0]
        cid = row[case_id_column]
        part = row[part_column]
        slide_to_case_part[slide_id] = (cid, part)
    
    # Group slides by (case_id, part)
    from collections import defaultdict
    groups = defaultdict(list)  # (case_id, part) -> list of indices into slide_identifiers
    for i, slide_id in enumerate(slide_identifiers):
        key = slide_to_case_part.get(slide_id)
        if key is None:
            continue
        groups[key].append(i)
    
    # Only case-parts that have at least one slide
    case_part_ids = []
    y_true_case = []
    y_proba_case = []
    case_part_metadata = []
    
    for (case_id, part_desc), indices in sorted(groups.items()):
        case_part_ids.append((case_id, part_desc))
        labels = raw_labels[indices]
        # All slides in same case-part should have same label (propagated from part)
        if len(np.unique(labels)) > 1:
            raise ValueError(
                f"Case-part ({case_id}, {part_desc}) has slides with different labels: {labels.tolist()}. "
                "Labels must be consistent per case-part."
            )
        y_true_case.append(int(labels[0]))
        
        probs = raw_probabilities[indices]  # (n_slides_in_group, n_classes)
        if case_level_method == 'mean':
            agg_probs = np.mean(probs, axis=0)
        else:
            agg_probs = np.max(probs, axis=0)
            # Renormalize so probabilities sum to 1
            agg_probs = agg_probs / agg_probs.sum()
        y_proba_case.append(agg_probs)
        
        case_part_metadata.append({
            'case_id': case_id,
            'part_description': part_desc,
            'n_slides': len(indices),
            'slide_ids': [slide_identifiers[j] for j in indices]
        })
    
    y_true_case = np.array(y_true_case)
    y_proba_case = np.array(y_proba_case)
    
    # Predictions: threshold for binary, argmax for multi-class
    if n_classes == 2 and optimal_threshold is not None:
        y_pred_case = (y_proba_case[:, 1] >= optimal_threshold).astype(int)
    else:
        y_pred_case = np.argmax(y_proba_case, axis=1)
    
    metrics = calculate_comprehensive_metrics(y_true_case, y_pred_case, y_proba_case, class_names)
    metrics['raw_labels'] = y_true_case
    metrics['raw_predictions'] = y_pred_case
    metrics['raw_probabilities'] = y_proba_case
    metrics['slide_identifiers'] = case_part_ids  # reuse key for downstream (case-part ids)
    
    return metrics, case_part_ids, case_part_metadata


def bootstrap_evaluation(y_true, y_proba, y_pred, slide_ids, class_names, n_iterations=1000, random_state=42):
    """
    Perform bootstrap evaluation to calculate confidence intervals for metrics.
    
    Args:
        y_true (list): True labels
        y_proba (np.array): Predicted probabilities
        y_pred (list): Predicted labels  
        slide_ids (list): Slide identifiers
        class_names (list): List of class names
        n_iterations (int): Number of bootstrap iterations
        random_state (int): Random seed for reproducibility
    
    Returns:
        dict: Dictionary containing bootstrap statistics and confidence intervals
    """
    print(f"\nPerforming bootstrap evaluation with {n_iterations} iterations...")
    
    np.random.seed(random_state)
    n_samples = len(y_true)
    is_binary = len(class_names) == 2
    
    # Storage for bootstrap statistics
    bootstrap_stats = {
        'accuracy': [],
        'roc_auc': [],
        'avg_precision': [],
        'precision_macro': [],
        'recall_macro': [],
        'f1_macro': []
    }
    
    # For binary classification, also track sensitivity and specificity
    if is_binary:
        bootstrap_stats['sensitivity'] = []
        bootstrap_stats['specificity'] = []
    
    # For binary classification, also store curve data for confidence intervals
    if is_binary:
        bootstrap_stats['roc_curves'] = []
        bootstrap_stats['pr_curves'] = []
        # Common FPR points for ROC curve interpolation
        common_fpr = np.linspace(0, 1, 100)
        # Common recall points for PR curve interpolation  
        common_recall = np.linspace(0, 1, 100)
    
    for i in range(n_iterations):
        if (i + 1) % 100 == 0:
            print(f"  Bootstrap iteration {i + 1}/{n_iterations}")
        
        # Bootstrap sample with replacement
        boot_indices = np.random.choice(n_samples, n_samples, replace=True)
        boot_y_true = [y_true[idx] for idx in boot_indices]
        boot_y_proba = y_proba[boot_indices]
        boot_y_pred = [y_pred[idx] for idx in boot_indices]
        
        # Calculate metrics for this bootstrap sample
        from modules import calculate_comprehensive_metrics
        boot_metrics = calculate_comprehensive_metrics(boot_y_true, boot_y_pred, boot_y_proba, class_names)
        
        # Store bootstrap statistics
        bootstrap_stats['accuracy'].append(boot_metrics['accuracy'])
        bootstrap_stats['roc_auc'].append(boot_metrics['roc_auc_macro'])
        bootstrap_stats['avg_precision'].append(boot_metrics['avg_precision_macro'])
        bootstrap_stats['precision_macro'].append(boot_metrics['precision_macro'])
        bootstrap_stats['recall_macro'].append(boot_metrics['recall_macro'])
        bootstrap_stats['f1_macro'].append(boot_metrics['f1_macro'])
        
        # For binary classification, store sensitivity and specificity
        if is_binary:
            bootstrap_stats['sensitivity'].append(boot_metrics.get('sensitivity', boot_metrics['recall_macro']))
            bootstrap_stats['specificity'].append(boot_metrics.get('specificity', 0.0))
        
        # For binary classification, store curve data
        if is_binary:
            # ROC curve
            fpr, tpr, _ = roc_curve(boot_y_true, boot_y_proba[:, 1])
            # Interpolate to common FPR points
            tpr_interp = np.interp(common_fpr, fpr, tpr)
            tpr_interp[0] = 0.0  # Ensure starts at (0,0)
            bootstrap_stats['roc_curves'].append(tpr_interp)
            
            # PR curve
            precision, recall, _ = precision_recall_curve(boot_y_true, boot_y_proba[:, 1])
            # Interpolate to common recall points (reverse order for PR curve)
            precision_interp = np.interp(common_recall, recall[::-1], precision[::-1])
            bootstrap_stats['pr_curves'].append(precision_interp)
    
    print("✓ Bootstrap evaluation completed")
    
    # Calculate confidence intervals (2.5th and 97.5th percentiles)
    confidence_intervals = {}
    for metric_name, values in bootstrap_stats.items():
        if metric_name not in ['roc_curves', 'pr_curves']:
            values = np.array(values)
            confidence_intervals[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'ci_lower': np.percentile(values, 2.5),
                'ci_upper': np.percentile(values, 97.5)
            }
    
    # For binary classification, calculate curve confidence intervals
    if is_binary:
        # ROC curve confidence intervals
        roc_curves = np.array(bootstrap_stats['roc_curves'])
        roc_mean = np.mean(roc_curves, axis=0)
        roc_ci_lower = np.percentile(roc_curves, 2.5, axis=0)
        roc_ci_upper = np.percentile(roc_curves, 97.5, axis=0)
        
        confidence_intervals['roc_curve'] = {
            'fpr': common_fpr,
            'tpr_mean': roc_mean,
            'tpr_ci_lower': roc_ci_lower,
            'tpr_ci_upper': roc_ci_upper
        }
        
        # PR curve confidence intervals
        pr_curves = np.array(bootstrap_stats['pr_curves'])
        pr_mean = np.mean(pr_curves, axis=0)
        pr_ci_lower = np.percentile(pr_curves, 2.5, axis=0)
        pr_ci_upper = np.percentile(pr_curves, 97.5, axis=0)
        
        confidence_intervals['pr_curve'] = {
            'recall': common_recall,
            'precision_mean': pr_mean,
            'precision_ci_lower': pr_ci_lower,
            'precision_ci_upper': pr_ci_upper
        }
    
    return {
        'bootstrap_stats': bootstrap_stats,
        'confidence_intervals': confidence_intervals,
        'n_iterations': n_iterations
    }


def get_color_palette(theme):
    """Get color palette based on theme."""
    palettes = {
        'red': {
            'primary': '#DC143C',      # Crimson
            'secondary': '#B22222',    # FireBrick
            'light': '#FFB6C1',       # LightPink
            'dark': '#8B0000',        # DarkRed
            'accent': '#FF6347'       # Tomato
        },
        'blue': {
            'primary': '#1E90FF',      # DodgerBlue
            'secondary': '#4169E1',    # RoyalBlue
            'light': '#87CEEB',       # SkyBlue
            'dark': '#191970',        # MidnightBlue
            'accent': '#00BFFF'       # DeepSkyBlue
        },
        'green': {
            'primary': '#32CD32',      # LimeGreen
            'secondary': '#228B22',    # ForestGreen
            'light': '#90EE90',       # LightGreen
            'dark': '#006400',        # DarkGreen
            'accent': '#00FF7F'       # SpringGreen
        },
        'purple': {
            'primary': '#8A2BE2',      # BlueViolet
            'secondary': '#9932CC',    # DarkOrchid
            'light': '#DDA0DD',       # Plum
            'dark': '#4B0082',        # Indigo
            'accent': '#BA55D3'       # MediumOrchid
        },
        'orange': {
            'primary': '#FF8C00',      # DarkOrange
            'secondary': '#FF6347',    # Tomato
            'light': '#FFE4B5',       # Moccasin
            'dark': '#FF4500',        # OrangeRed
            'accent': '#FFA500'       # Orange
        }
    }
    return palettes.get(theme, palettes['blue'])


def set_matplotlib_font(font_family='DejaVu Sans'):
    """Set matplotlib font configuration."""
    import matplotlib.pyplot as plt
    import matplotlib
    
    # Set the font family
    plt.rcParams['font.family'] = font_family
    
    # If DejaVu Sans is not available, fallback to sans-serif
    try:
        matplotlib.font_manager.findfont(font_family, fallback_to_default=False)
    except:
        print(f"Warning: Font '{font_family}' not found, using default sans-serif")
        plt.rcParams['font.family'] = 'sans-serif'


def style_plot(ax, fontsize=12, font_family='DejaVu Sans', remove_spines=True):
    """Apply consistent styling to plots."""
    # Remove top and right spines
    if remove_spines:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Set font sizes and family
    ax.tick_params(labelsize=fontsize-2)
    ax.xaxis.label.set_fontsize(fontsize)
    ax.yaxis.label.set_fontsize(fontsize)
    ax.xaxis.label.set_fontfamily(font_family)
    ax.yaxis.label.set_fontfamily(font_family)
    
    if ax.get_title():
        ax.set_title(ax.get_title(), fontsize=fontsize+2, fontweight='bold', fontfamily=font_family)
    
    # Set font for tick labels
    for label in ax.get_xticklabels():
        label.set_fontfamily(font_family)
    for label in ax.get_yticklabels():
        label.set_fontfamily(font_family)
    
    # Handle legend font size if legend exists
    if ax.get_legend():
        legend = ax.get_legend()
        for text in legend.get_texts():
            text.set_fontfamily(font_family)
            text.set_fontsize(fontsize)
    
    # # Grid styling
    # ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    # ax.set_axisbelow(True)


def plot_binary_classification_metrics(metrics, bootstrap_results=None, class_names=None, 
                                      color_theme='blue', fontsize=12, font_family='DejaVu Sans', 
                                      output_path=None):
    """
    Plot binary classification metrics as a bar chart with optional confidence intervals.
    
    Args:
        metrics (dict): Evaluation metrics
        bootstrap_results (dict, optional): Bootstrap results for confidence intervals
        class_names (list, optional): Class names for labeling
        color_theme (str): Color theme to use
        fontsize (int): Font size for text
        font_family (str): Font family to use
        output_path (str, optional): Path to save the plot
    
    Returns:
        None
    """
    import matplotlib.pyplot as plt
    
    # Get color palette
    colors = get_color_palette(color_theme)
    
    # Define metrics to plot: Accuracy, Sensitivity, Specificity, F1-Score
    metric_names = ['Accuracy', 'Sensitivity', 'Specificity', 'F1-Score']
    
    # Get metric values, with fallbacks for sensitivity and specificity
    accuracy = metrics['accuracy']
    sensitivity = metrics.get('sensitivity', metrics.get('recall_macro', 0.0))
    specificity = metrics.get('specificity', 0.0)
    f1_score = metrics['f1_macro']
    
    metric_keys = ['accuracy', 'sensitivity', 'specificity', 'f1_macro']
    values = [accuracy, sensitivity, specificity, f1_score]
    
    # Get confidence intervals if bootstrap results are available
    ci_lower = None
    ci_upper = None
    if bootstrap_results:
        ci_lower = []
        ci_upper = []
        for i, key in enumerate(metric_keys):
            if key in bootstrap_results['confidence_intervals']:
                ci_lower.append(bootstrap_results['confidence_intervals'][key]['ci_lower'])
                ci_upper.append(bootstrap_results['confidence_intervals'][key]['ci_upper'])
            else:
                # Fallback: use point estimate (no CI)
                ci_lower.append(values[i])
                ci_upper.append(values[i])
        errors = [[values[i] - ci_lower[i] for i in range(len(values))],
                 [ci_upper[i] - values[i] for i in range(len(values))]]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bars
    x_pos = np.arange(len(metric_names))
    bars = ax.bar(x_pos, values, color=colors['primary'], alpha=0.8, 
                  edgecolor=colors['dark'], linewidth=1.5)
    
    # Add error bars if bootstrap results are available
    if bootstrap_results:
        ax.errorbar(x_pos, values, yerr=errors, fmt='none', 
                   color=colors['dark'], capsize=5, capthick=2, linewidth=2)
    
    # Add value text and CI text above the error bar to avoid clashing with the bar
    for i, (bar, value) in enumerate(zip(bars, values)):
        height = bar.get_height()
        bar_center = bar.get_x() + bar.get_width()/2.
        if bootstrap_results:
            # Show value and CI, with CI above the error bar
            value_text = f"{value:.3f}"
            ci_text = f"[{ci_lower[i]:.3f}, {ci_upper[i]:.3f}]"
            y_offset = 0.01
            ci_y_offset = 0.01 + (errors[1][i] if errors[1][i] > 0 else 0)  # place above error bar
            # # Value text just above bar
            # ax.text(bar_center, height + ci_y_offset,
            #         value_text, ha='center', va='bottom', fontsize=fontsize-1, 
            #         fontweight='bold')
            # CI text above error bar
            ax.text(bar_center, height + ci_y_offset,
                    value_text+"\n"+ci_text, ha='center', va='bottom', fontsize=fontsize-3, 
                    fontweight='normal', fontfamily=font_family)
        else:
            text = f"{value:.3f}"
            y_offset = 0.01
            ax.text(bar_center, height + y_offset,
                    text, ha='center', va='bottom', fontsize=fontsize-1, 
                    fontweight='bold', fontfamily=font_family)
    
    # Customize the plot
    # ax.set_xlabel('Metrics', fontsize=fontsize, fontfamily=font_family)
    ax.set_ylabel('Score', fontsize=fontsize, fontfamily=font_family)
    
    # Create title
    if class_names and len(class_names) == 2:
        title = f'Binary Classification Metrics\n({class_names[1]} vs {class_names[0]})'
    else:
        title = 'Binary Classification Metrics'
    
    if bootstrap_results:
        title += f'\nwith 95% Confidence Intervals ({bootstrap_results["n_iterations"]} bootstrap iterations)'
    
    # ax.set_title(title, fontsize=fontsize+2, fontweight='bold', fontfamily=font_family, pad=20)
    
    # Set x-axis
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metric_names, fontsize=fontsize, fontfamily=font_family)
    
    # Set y-axis limits
    y_max = max(values) if not bootstrap_results else max(ci_upper)
    ax.set_ylim(0, min(1.0, y_max + 0.15))
    
    # # Add horizontal line at 0.5 for reference
    # ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, linewidth=1)
    # ax.text(0.02, 0.52, 'Random Performance', transform=ax.transAxes, 
    #         fontsize=fontsize-2, color='gray', style='italic')
    
    # Apply styling
    style_plot(ax, fontsize, font_family, remove_spines=True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def analyze_test_set_distribution(test_df, class_names):
    """Analyze and print test set class distribution."""
    print(f"\nTest Set Analysis:")
    print(f"Total samples: {len(test_df)}")
    print(f"Unique patients: {test_df['MRN'].nunique()}")
    print(f"Unique slides: {test_df['slide_name'].nunique()}")
    
    # Class distribution
    print(f"\nClass Distribution:")
    class_counts = test_df['diagnosis'].value_counts().sort_index()
    for diagnosis, count in class_counts.items():
        percentage = (count / len(test_df)) * 100
        print(f"  {diagnosis}: {count} samples ({percentage:.1f}%)")
    
    # Check for patients with multiple slides
    patient_slide_counts = test_df.groupby('MRN').size()
    patients_with_multiple_slides = (patient_slide_counts > 1).sum()
    max_slides_per_patient = patient_slide_counts.max()
    
    print(f"\nPatient-Slide Relationship:")
    print(f"Patients with multiple slides: {patients_with_multiple_slides}")
    print(f"Maximum slides per patient: {max_slides_per_patient}")
    
    # Verify all classes in test set are in model classes
    test_classes = set(test_df['diagnosis'].unique())
    model_classes = set(class_names)
    
    if not test_classes.issubset(model_classes):
        missing_classes = test_classes - model_classes
        print(f"⚠️  Warning: Test set contains classes not in model: {missing_classes}")
    
    if not model_classes.issubset(test_classes):
        unused_classes = model_classes - test_classes
        print(f"ℹ️  Info: Model classes not in test set: {unused_classes}")
    
    print()


def plot_roc_curve_with_ci(y_true, y_proba, class_names, title, output_path, bootstrap_results=None, 
                          color_theme='blue', fontsize=12, font_family='DejaVu Sans'):
    """
    Plot ROC curve for binary classification with optional confidence intervals.
    
    Args:
        y_true: True binary labels (0, 1)
        y_proba: Predicted probabilities for the positive class
        class_names: List of class names [negative_class, positive_class]
        title: Plot title
        output_path: Output file path
        bootstrap_results: Optional bootstrap results for confidence intervals
        color_theme: Color theme to use
        fontsize: Font size for text
        font_family: Font family to use
    
    Returns:
        float: ROC AUC score
    """
    import matplotlib.pyplot as plt
    
    # Get color palette
    colors = get_color_palette(color_theme)
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(6, 6))


    # Plot main ROC curve
    ax.plot(fpr, tpr, color=colors['primary'], lw=3, 
            label=f'ROC curve (AUC = {roc_auc:.3f})')
    
    # Plot confidence interval if bootstrap results are provided
    if bootstrap_results and 'roc_curve' in bootstrap_results['confidence_intervals']:
        ci_data = bootstrap_results['confidence_intervals']['roc_curve']
        # If available, get the AUC CI for the legend
        auc_ci_str = ""
        if bootstrap_results and 'roc_auc' in bootstrap_results['confidence_intervals']:
            auc_ci = bootstrap_results['confidence_intervals']['roc_auc']
            auc_ci_str = f" [{auc_ci['ci_lower']:.3f}, {auc_ci['ci_upper']:.3f}]"
        ax.fill_between(
            ci_data['fpr'],
            ci_data['tpr_ci_lower'],
            ci_data['tpr_ci_upper'],
            alpha=0.2,
            color=colors['primary'],
            label=f'95% CI{auc_ci_str}'
        )

    ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', 
            label='Random classifier')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontfamily=font_family)
    ax.set_ylabel('True Positive Rate', fontfamily=font_family)
    # ax.set_title(title, fontfamily=font_family)
    
    # Apply styling first
    style_plot(ax, fontsize, font_family, remove_spines=True)
    
    # Set legend with proper font size after styling
    legend = ax.legend(loc="lower right", fontsize=fontsize-1, prop={'family': font_family, 'size': fontsize})
    # Ensure legend text uses the correct font family and size
    for text in legend.get_texts():
        text.set_fontfamily(font_family)
        text.set_fontsize(fontsize)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return roc_auc


def plot_precision_recall_curve_with_ci(y_true, y_proba, class_names, title, output_path, bootstrap_results=None,
                                       color_theme='blue', fontsize=12, font_family='DejaVu Sans'):
    """
    Plot Precision-Recall curve for binary classification with optional confidence intervals.
    
    Args:
        y_true: True binary labels (0, 1)
        y_proba: Predicted probabilities for the positive class
        class_names: List of class names [negative_class, positive_class]
        title: Plot title
        output_path: Output file path
        bootstrap_results: Optional bootstrap results for confidence intervals
        color_theme: Color theme to use
        fontsize: Font size for text
        font_family: Font family to use
    
    Returns:
        float: Average Precision (AUPRC) score
    """
    import matplotlib.pyplot as plt
    
    # Get color palette
    colors = get_color_palette(color_theme)
    
    # Calculate Precision-Recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    average_precision = average_precision_score(y_true, y_proba)
    
    # Calculate baseline (random classifier performance)
    positive_ratio = np.mean(y_true)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(6, 6))
    

    
    # Plot main PR curve
    ax.plot(recall, precision, color=colors['primary'], lw=3, 
            label=f'PR curve (AP = {average_precision:.3f})')
    # Plot confidence interval if bootstrap results are provided
    if bootstrap_results and 'pr_curve' in bootstrap_results['confidence_intervals']:
        ap_ci = bootstrap_results['confidence_intervals']['avg_precision']
        ci_data = bootstrap_results['confidence_intervals']['pr_curve']
        ax.fill_between(ci_data['recall'], ci_data['precision_ci_lower'], ci_data['precision_ci_upper'], 
                       alpha=0.2, color=colors['primary'], label=f'95% CI [{ap_ci["ci_lower"]:.3f}, {ap_ci["ci_upper"]:.3f}]')
    ax.axhline(y=positive_ratio, color='gray', lw=2, linestyle='--', 
               label=f'Random classifier (AP = {positive_ratio:.3f})')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall (Sensitivity)', fontfamily=font_family)
    ax.set_ylabel('Precision', fontfamily=font_family)
    # ax.set_title(title, fontfamily=font_family)
    
    # Apply styling first
    style_plot(ax, fontsize, font_family, remove_spines=True)
    
    # Set legend with proper font size after styling
    legend = ax.legend(loc="lower left", fontsize=fontsize-1, prop={'family': font_family, 'size': fontsize})
    # Ensure legend text uses the correct font family and size
    for text in legend.get_texts():
        text.set_fontfamily(font_family)
        text.set_fontsize(fontsize)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return average_precision


def print_bootstrap_results(bootstrap_results):
    """Print bootstrap confidence interval results."""
    print("\n" + "=" * 80)
    print("BOOTSTRAP CONFIDENCE INTERVALS (95%)")
    print("=" * 80)
    
    ci = bootstrap_results['confidence_intervals']
    n_iter = bootstrap_results['n_iterations']
    
    print(f"Number of bootstrap iterations: {n_iter}")
    print("\nMetric                        Mean ± SD        [95% CI]")
    print("-" * 65)
    
    # Standard metrics
    for metric_name in ['accuracy', 'roc_auc', 'avg_precision', 'precision_macro', 'recall_macro', 'f1_macro']:
        if metric_name in ci:
            stats = ci[metric_name]
            print(f"{metric_name:<25} {stats['mean']:.4f} ± {stats['std']:.4f}   [{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]")
    
    # Binary classification specific metrics (sensitivity and specificity)
    for metric_name in ['sensitivity', 'specificity']:
        if metric_name in ci:
            stats = ci[metric_name]
            print(f"{metric_name:<25} {stats['mean']:.4f} ± {stats['std']:.4f}   [{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]")
    
    print("\n" + "=" * 80)


def print_evaluation_results(metrics, class_names, bootstrap_results=None, level_label=''):
    """Print comprehensive evaluation results."""
    is_binary_classification = len(class_names) == 2
    
    print("\n" + "=" * 80)
    print("TEST SET EVALUATION RESULTS" + (level_label if level_label else ""))
    print("=" * 80)
    
    # Print overall metrics
    print("OVERALL PERFORMANCE METRICS:")
    
    def format_metric_with_ci(metric_name, point_est, bootstrap_results=None):
        """Format metric with optional confidence interval."""
        if bootstrap_results and metric_name in bootstrap_results['confidence_intervals']:
            ci = bootstrap_results['confidence_intervals'][metric_name]
            return f"{point_est:.4f} [95% CI: {ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]"
        else:
            return f"{point_est:.4f}"
    
    print(f"  Accuracy:                    {format_metric_with_ci('accuracy', metrics['accuracy'], bootstrap_results)}")
    print(f"  Precision (Macro):           {format_metric_with_ci('precision_macro', metrics['precision_macro'], bootstrap_results)}")
    print(f"  Recall (Macro):              {format_metric_with_ci('recall_macro', metrics['recall_macro'], bootstrap_results)}")
    print(f"  F1-Score (Macro):            {format_metric_with_ci('f1_macro', metrics['f1_macro'], bootstrap_results)}")
    
    if not is_binary_classification:
        print(f"  Precision (Micro):           {metrics['precision_micro']:.4f}")
        print(f"  Recall (Micro):              {metrics['recall_micro']:.4f}")
        print(f"  F1-Score (Micro):            {metrics['f1_micro']:.4f}")
        print(f"  ROC AUC (Macro):             {format_metric_with_ci('roc_auc', metrics['roc_auc_macro'], bootstrap_results)}")
        print(f"  ROC AUC (Micro):             {metrics['roc_auc_micro']:.4f}")
        print(f"  Average Precision (Macro):   {format_metric_with_ci('avg_precision', metrics['avg_precision_macro'], bootstrap_results)}")
        print(f"  Average Precision (Micro):   {metrics['avg_precision_micro']:.4f}")
    else:
        # Binary classification specific metrics
        print(f"  Sensitivity (Recall):        {format_metric_with_ci('recall_macro', metrics.get('sensitivity', metrics['recall_macro']), bootstrap_results)}")
        print(f"  Specificity:                 {metrics.get('specificity', 'N/A')}")
        print(f"  ROC AUC:                     {format_metric_with_ci('roc_auc', metrics['roc_auc_macro'], bootstrap_results)}")
        print(f"  Average Precision:           {format_metric_with_ci('avg_precision', metrics['avg_precision_macro'], bootstrap_results)}")
        print(f"  True Positives:              {metrics.get('true_positives', 'N/A')}")
        print(f"  True Negatives:              {metrics.get('true_negatives', 'N/A')}")
        print(f"  False Positives:             {metrics.get('false_positives', 'N/A')}")
        print(f"  False Negatives:             {metrics.get('false_negatives', 'N/A')}")
    
    print(f"  Matthews Correlation Coeff:  {metrics['mcc']:.4f}")
    
    # Print per-class metrics
    print(f"\nPER-CLASS PERFORMANCE:")
    print(f"{'Class':<25} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 60)
    for _, row in metrics['per_class_metrics'].iterrows():
        print(f"{row['class']:<25} {row['precision']:<10.4f} {row['recall']:<10.4f} {row['f1']:<10.4f}")


def create_evaluation_visualizations(metrics, class_names, output_dir, bootstrap_results=None, 
                                   color_theme='blue', fontsize=12, font_family='DejaVu Sans', output_suffix=''):
    """Create visualizations for evaluation metrics."""
    print(f"\nCreating evaluation visualizations...")
    
    # Set matplotlib font configuration
    set_matplotlib_font(font_family)
    
    level_label = " (Case-Part Level)" if output_suffix else ""
    
    # Plot confusion matrix
    plot_confusion_matrix(
        metrics['confusion_matrix'], class_names,
        f'Confusion Matrix - Test Set{level_label}',
        os.path.join(output_dir, f'confusion_matrix{output_suffix}.png'),
        color_theme=color_theme, fontsize=fontsize
    )
    print("✓ Confusion matrix saved")
    
    # Plot per-class metrics
    plot_per_class_metrics(
        metrics['per_class_metrics'].set_index('class'),
        f'Per-Class Performance Metrics - Test Set{level_label}',
        os.path.join(output_dir, f'per_class_metrics{output_suffix}.png'),
        color_theme=color_theme, fontsize=fontsize
    )
    print("✓ Per-class metrics plot saved")
    
    # For binary classification, create additional visualizations
    is_binary_classification = len(class_names) == 2
    if is_binary_classification:
        bootstrap_suffix = " with Bootstrap CI" if bootstrap_results else ""
        print(f"Creating binary classification visualizations{bootstrap_suffix}...")
        
        # Extract positive class probabilities
        y_true = metrics['raw_labels']
        y_proba = metrics['raw_probabilities'][:, 1]  # Probability of positive class (class 1)
        
        # Binary Classification Metrics Bar Chart
        plot_binary_classification_metrics(
            metrics, bootstrap_results, class_names,
            color_theme=color_theme, fontsize=fontsize, font_family=font_family,
            output_path=os.path.join(output_dir, f'binary_metrics_summary{output_suffix}.png')
        )
        print("✓ Binary classification metrics bar chart saved")
        
        # ROC Curve with optional confidence intervals
        roc_auc = plot_roc_curve_with_ci(
            y_true, y_proba, class_names,
            f'ROC Curve - Test Set ({class_names[1]} vs {class_names[0]}){level_label}{bootstrap_suffix}',
            os.path.join(output_dir, f'roc_curve{output_suffix}.png'),
            bootstrap_results, color_theme, fontsize, font_family
        )
        
        ci_text = ""
        if bootstrap_results and 'roc_auc' in bootstrap_results['confidence_intervals']:
            auc_ci = bootstrap_results['confidence_intervals']['roc_auc']
            ci_text = f" [95% CI: {auc_ci['ci_lower']:.3f}-{auc_ci['ci_upper']:.3f}]"
        print(f"✓ ROC curve saved with AUC = {roc_auc:.3f}{ci_text}")
        
        # Precision-Recall Curve with optional confidence intervals
        average_precision = plot_precision_recall_curve_with_ci(
            y_true, y_proba, class_names,
            f'Precision-Recall Curve - Test Set ({class_names[1]} vs {class_names[0]}){level_label}{bootstrap_suffix}',
            os.path.join(output_dir, f'precision_recall_curve{output_suffix}.png'),
            bootstrap_results, color_theme, fontsize, font_family
        )
        
        ci_text = ""
        if bootstrap_results and 'avg_precision' in bootstrap_results['confidence_intervals']:
            ap_ci = bootstrap_results['confidence_intervals']['avg_precision']
            ci_text = f" [95% CI: {ap_ci['ci_lower']:.3f}-{ap_ci['ci_upper']:.3f}]"
        print(f"✓ PRC curve saved with AP = {average_precision:.3f}{ci_text}")
        
        # Compare ROC AUC vs AUPRC for imbalanced data insight
        positive_ratio = np.mean(y_true)
        print(f"📊 Binary Classification Curve Comparison:")
        print(f"   Positive class prevalence: {positive_ratio:.1%}")
        print(f"   ROC AUC: {roc_auc:.3f} (focus: TPR vs FPR)")
        print(f"   AUPRC:   {average_precision:.3f} (focus: Precision vs Recall)")
        if positive_ratio < 0.3:  # Imbalanced dataset
            print(f"   💡 For imbalanced data: AUPRC is more informative than ROC AUC")
        if average_precision > roc_auc:
            print(f"   ✅ AUPRC > ROC AUC suggests good precision at high recall")
        elif roc_auc > average_precision + 0.1:
            print(f"   ⚠️  ROC AUC >> AUPRC may indicate precision challenges")


def save_evaluation_results(metrics, class_names, test_dataloader, model_config, output_dir, args, bootstrap_results=None,
                            case_level=False, case_part_metadata=None, optimal_threshold=None, operating_point_method=None,
                            raw_slide_identifiers=None, raw_slide_probabilities=None,
                            raw_slide_case_ids=None, raw_slide_part_descriptions=None):
    """Save detailed evaluation results to JSON and CSV files."""
    print(f"\nSaving detailed evaluation results...")
    
    is_binary_classification = len(class_names) == 2
    
    # Total samples: case-parts when case_level, else slides
    total_test_samples = len(metrics['raw_labels'])
    
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
    
    evaluation_config = {
        'test_csv_path': args.test_csv_path,
        'model_path': args.model_path,
        'h5_directory': args.h5_directory,
        'batch_size': args.batch_size,
        'device': str(args.device),
        'perform_bootstrapping': args.perform_bootstrapping,
        'bootstrap_iterations': args.bootstrap_iterations if args.perform_bootstrapping else None
    }

    # Operating-point details used for binary threshold-based prediction.
    # Note: if `optimal_threshold` is None, the code falls back to argmax (so `threshold_used` stays None).
    evaluation_config['operating_point'] = {
        'threshold_from_checkpoint': float(optimal_threshold) if optimal_threshold is not None else None,
        'threshold_used': float(optimal_threshold) if (is_binary_classification and optimal_threshold is not None) else None,
        'method': operating_point_method
    }
    if case_level:
        evaluation_config['case_level'] = True
        evaluation_config['case_level_method'] = args.case_level_method
        evaluation_config['case_id_column'] = args.case_id_column
        evaluation_config['part_column'] = args.part_column
    
    # Create results dictionary
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
        'total_test_samples': total_test_samples,
        'model_config': to_json_serializable(model_config),
        'evaluation_config': evaluation_config
    }
    
    # Add bootstrap results if available
    if bootstrap_results:
        results['bootstrap_evaluation'] = {
            'confidence_intervals': {
                metric: {
                    'mean': float(stats['mean']),
                    'std': float(stats['std']),
                    'ci_lower': float(stats['ci_lower']),
                    'ci_upper': float(stats['ci_upper'])
                } for metric, stats in bootstrap_results['confidence_intervals'].items()
                if metric not in ['roc_curve', 'pr_curve']
            },
            'n_iterations': bootstrap_results['n_iterations']
        }
        
        # Include curve confidence intervals for binary classification
        if is_binary_classification and 'roc_curve' in bootstrap_results['confidence_intervals']:
            roc_ci = bootstrap_results['confidence_intervals']['roc_curve']
            pr_ci = bootstrap_results['confidence_intervals']['pr_curve']
            
            results['bootstrap_evaluation']['curve_confidence_intervals'] = {
                'roc_curve': {
                    'fpr': roc_ci['fpr'].tolist(),
                    'tpr_mean': roc_ci['tpr_mean'].tolist(),
                    'tpr_ci_lower': roc_ci['tpr_ci_lower'].tolist(),
                    'tpr_ci_upper': roc_ci['tpr_ci_upper'].tolist()
                },
                'pr_curve': {
                    'recall': pr_ci['recall'].tolist(),
                    'precision_mean': pr_ci['precision_mean'].tolist(),
                    'precision_ci_lower': pr_ci['precision_ci_lower'].tolist(),
                    'precision_ci_upper': pr_ci['precision_ci_upper'].tolist()
                }
            }
    
    # Save results to JSON
    with open(os.path.join(output_dir, 'test_evaluation_results.json'), 'w') as f:
        json.dump(to_json_serializable(results), f, indent=2)
    print("✓ Results saved to test_evaluation_results.json")
    
    # Save per-class metrics as CSV
    metrics['per_class_metrics'].to_csv(
        os.path.join(output_dir, 'per_class_metrics.csv'), index=False
    )
    print("✓ Per-class metrics saved to per_class_metrics.csv")
    
    # Save individual predictions if requested
    if args.save_predictions:
        ids = metrics['slide_identifiers']
        if case_level and case_part_metadata is not None:
            # Case-part level: case_id, part_description, n_slides, ...
            predictions_df = pd.DataFrame({
                'case_id': [cp[0] for cp in ids],
                'part_description': [cp[1] for cp in ids],
                'n_slides': [m['n_slides'] for m in case_part_metadata],
                'true_label': metrics['raw_labels'],
                'predicted_label': metrics['raw_predictions'],
                'true_class': [class_names[i] for i in metrics['raw_labels']],
                'predicted_class': [class_names[i] for i in metrics['raw_predictions']]
            })
        else:
            predictions_df = pd.DataFrame({
                'slide_id': ids,
                'true_label': metrics['raw_labels'],
                'predicted_label': metrics['raw_predictions'],
                'true_class': [class_names[i] for i in metrics['raw_labels']],
                'predicted_class': [class_names[i] for i in metrics['raw_predictions']]
            })
        
        # Add probability columns
        for i, class_name in enumerate(class_names):
            predictions_df[f'prob_{class_name}'] = metrics['raw_probabilities'][:, i]
        
        predictions_df.to_csv(
            os.path.join(output_dir, 'individual_predictions.csv'), index=False
        )
        print("✓ Individual predictions saved to individual_predictions.csv")

        # When running at case-part level, also save the raw per-slide probabilities.
        # This is useful because `metrics['raw_probabilities']` has been aggregated to case-part level.
        if case_level and raw_slide_identifiers is not None and raw_slide_probabilities is not None:
            if len(raw_slide_identifiers) != raw_slide_probabilities.shape[0]:
                raise ValueError(
                    f"raw_slide_identifiers length ({len(raw_slide_identifiers)}) does not match "
                    f"raw_slide_probabilities shape[0] ({raw_slide_probabilities.shape[0]})."
                )

            slide_probs_df = pd.DataFrame({args.slide_id_column: raw_slide_identifiers})

            if raw_slide_case_ids is not None:
                if len(raw_slide_case_ids) != raw_slide_probabilities.shape[0]:
                    raise ValueError(
                        f"raw_slide_case_ids length ({len(raw_slide_case_ids)}) does not match "
                        f"raw_slide_probabilities shape[0] ({raw_slide_probabilities.shape[0]})."
                    )
                slide_probs_df['case_id'] = raw_slide_case_ids

            if raw_slide_part_descriptions is not None:
                if len(raw_slide_part_descriptions) != raw_slide_probabilities.shape[0]:
                    raise ValueError(
                        f"raw_slide_part_descriptions length ({len(raw_slide_part_descriptions)}) does not match "
                        f"raw_slide_probabilities shape[0] ({raw_slide_probabilities.shape[0]})."
                    )
                slide_probs_df['part_description'] = raw_slide_part_descriptions

            for i, class_name in enumerate(class_names):
                slide_probs_df[f'prob_{class_name}'] = raw_slide_probabilities[:, i]

            slide_probs_df.to_csv(
                os.path.join(output_dir, 'raw_slide_probabilities.csv'),
                index=False
            )
            print("✓ Raw slide probabilities saved to raw_slide_probabilities.csv")


def main():
    """Main evaluation function."""
    args = parse_arguments()
    
    # Set seed for reproducibility
    set_seed(args.random_state)
    
    print("=" * 80)
    print("TEST SET EVALUATION")
    print("=" * 80)
    print("Evaluating trained model on test set")
    print("=" * 80)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine device
    if args.device == 'auto':
        device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load trained model
    model, model_config, class_names, optimal_threshold, operating_point_method = load_trained_model(
        args.model_path,
        device,
        override_class_names=args.override_class_names,
        strict_checkpoint=args.strict_checkpoint
    )
    
    # Create test dataloader (class_names from checkpoint -> stable label ordering)
    test_dataloader, test_df, slide_identifiers = create_test_dataloader(
        args.test_csv_path, args.h5_directory,
        args.batch_size, args.num_workers, args.random_state, args.slide_id_column,
        class_names=class_names
    )
    
    # Analyze test set if requested
    if args.analyze_class_distribution:
        analyze_test_set_distribution(test_df, class_names)
        if args.case_level and args.case_id_column in test_df.columns and args.part_column in test_df.columns:
            # Case-part level summary
            case_part_df = test_df.groupby([args.case_id_column, args.part_column], as_index=False).agg(
                n_slides=('diagnosis', 'count'),
                diagnosis=('diagnosis', 'first')
            )
            print(f"\nCase-Part Level (for --case_level evaluation):")
            print(f"  Unique case-parts: {len(case_part_df)}")
            print(f"  Slides per case-part: min={case_part_df['n_slides'].min()}, max={case_part_df['n_slides'].max()}, mean={case_part_df['n_slides'].mean():.2f}")
            print(f"  Class distribution (one label per case-part):")
            for diagnosis, count in case_part_df['diagnosis'].value_counts().sort_index().items():
                pct = 100.0 * count / len(case_part_df)
                print(f"    {diagnosis}: {count} case-parts ({pct:.1f}%)")
    
    # Evaluate model
    print("\nEvaluating model on test set...")
    metrics = evaluate_model_with_identifiers(model, test_dataloader, class_names, slide_identifiers, device, optimal_threshold)
    
    # Stash raw per-slide outputs so we can optionally save them even after aggregating to case-part level.
    raw_slide_identifiers = metrics.get('slide_identifiers')
    raw_slide_probabilities = metrics.get('raw_probabilities')

    raw_slide_case_ids = None
    raw_slide_part_descriptions = None
    if args.case_level:
        # Map each slide_id to its case_id and part description.
        # The output column names must match `individual_predictions.csv` in case_level mode.
        needed_cols = [args.slide_id_column, args.case_id_column, args.part_column]
        missing_cols = [c for c in needed_cols if c not in test_df.columns]
        if missing_cols:
            raise ValueError(
                "Case-level raw slide probability export requires these columns in the test CSV: "
                f"{missing_cols} missing."
            )

        slide_meta = (
            test_df[[args.slide_id_column, args.case_id_column, args.part_column]]
            .drop_duplicates(subset=[args.slide_id_column])
            .set_index(args.slide_id_column)
        )

        # Preserve dataloader order (raw_slide_identifiers).
        raw_slide_case_ids = slide_meta[args.case_id_column].reindex(raw_slide_identifiers).tolist()
        raw_slide_part_descriptions = slide_meta[args.part_column].reindex(raw_slide_identifiers).tolist()

    case_part_metadata = None
    if args.case_level:
        # Validate CSV has case-level columns
        for col, name in [(args.case_id_column, 'case_id_column'), (args.part_column, 'part_column')]:
            if col not in test_df.columns:
                raise ValueError(
                    f"Case-level evaluation requires column '{col}' ({name}) in test CSV. "
                    f"Available columns: {list(test_df.columns)}"
                )
        print("\nAggregating slide-level results to case-part level...")
        metrics, case_part_ids, case_part_metadata = aggregate_slide_results_to_case_level(
            metrics['slide_identifiers'],
            metrics['raw_labels'],
            metrics['raw_probabilities'],
            test_df,
            args.slide_id_column,
            args.case_id_column,
            args.part_column,
            class_names,
            args.case_level_method,
            optimal_threshold=optimal_threshold,
        )
        print(f"  Aggregated {len(metrics['slide_identifiers'])} case-parts (method={args.case_level_method})")
    
    # Perform bootstrap evaluation if requested
    bootstrap_results = None
    if args.perform_bootstrapping:
        bootstrap_results = bootstrap_evaluation(
            metrics['raw_labels'], 
            metrics['raw_probabilities'], 
            metrics['raw_predictions'],
            metrics['slide_identifiers'],
            class_names,
            n_iterations=args.bootstrap_iterations,
            random_state=args.random_state
        )
    
    # Print evaluation results
    level_label = " (Case-Part Level)" if args.case_level else ""
    print_evaluation_results(metrics, class_names, bootstrap_results, level_label=level_label)
    
    # Print bootstrap results if available
    if bootstrap_results:
        print_bootstrap_results(bootstrap_results)
    
    # Create evaluation visualizations
    output_suffix = '_case_level' if args.case_level else ''
    create_evaluation_visualizations(metrics, class_names, args.output_dir, bootstrap_results,
                                    args.color_theme, args.fontsize, args.font_family, output_suffix=output_suffix)
    
    # Save detailed results
    save_evaluation_results(
        metrics, class_names, test_dataloader, model_config, args.output_dir, args, bootstrap_results,
        case_level=args.case_level, case_part_metadata=case_part_metadata,
        optimal_threshold=optimal_threshold, operating_point_method=operating_point_method,
        raw_slide_identifiers=raw_slide_identifiers if args.case_level else None,
        raw_slide_probabilities=raw_slide_probabilities if args.case_level else None,
        raw_slide_case_ids=raw_slide_case_ids if args.case_level else None,
        raw_slide_part_descriptions=raw_slide_part_descriptions if args.case_level else None
    )
    
    # Final summary
    print(f"\n" + "=" * 80)
    print("TEST EVALUATION COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    classification_type = "Binary" if len(class_names) == 2 else "Multi-class"
    print(f"✓ {classification_type} model evaluated on {len(class_names)} classes")
    sample_label = "case-parts" if args.case_level else "samples"
    n_eval = len(metrics['raw_labels'])
    print(f"✓ Test set size: {n_eval} {sample_label}")
    print(f"✓ Comprehensive evaluation metrics calculated")
    
    if args.perform_bootstrapping:
        print(f"✓ Bootstrap evaluation completed ({args.bootstrap_iterations} iterations)")
        print(f"✓ 95% confidence intervals calculated for all metrics")
    
    print(f"✓ Results saved to: {args.output_dir}")
    print(f"✓ Test accuracy: {metrics['accuracy']:.4f}")
    print(f"✓ Test F1-Score (Macro): {metrics['f1_macro']:.4f}")
    if len(class_names) == 2:
        print(f"✓ ROC AUC: {metrics['roc_auc_macro']:.4f}")
        print(f"✓ AUPRC: {metrics['avg_precision_macro']:.4f}")
        bootstrap_suffix = " (with 95% CI)" if args.perform_bootstrapping else ""
        fig_suffix = output_suffix  # '_case_level' or ''
        print(f"✓ Binary metrics summary{bootstrap_suffix} saved to: {os.path.join(args.output_dir, f'binary_metrics_summary{fig_suffix}.png')}")
        print(f"✓ ROC curve{bootstrap_suffix} saved to: {os.path.join(args.output_dir, f'roc_curve{fig_suffix}.png')}")
        print(f"✓ PRC curve{bootstrap_suffix} saved to: {os.path.join(args.output_dir, f'precision_recall_curve{fig_suffix}.png')}")
    print("=" * 80)

if __name__ == "__main__":
    main()
