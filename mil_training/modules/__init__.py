# Core Biopsy AI modules package

from .feature_dataset import FeatureDataset, create_patient_level_dataloaders, create_test_only_dataloader, collate_fn
from .gated_attn_pool import GatedMILClassifier
from .metrics import calculate_binary_metrics, calculate_comprehensive_metrics
from .visualization import (
    plot_confusion_matrix, plot_per_class_metrics, 
    plot_training_metrics, plot_roc_curve, plot_precision_recall_curve
)
from .training import train_model, evaluate_model
from .utils import set_seed, to_json_serializable
from .logger import setup_logger, get_logger, log_args, log_section
from .balanced_training import (
    BalancedTrainingConfig, calculate_class_weights, create_weighted_sampler,
    get_balanced_loss_function, analyze_class_distribution,
    get_metric_recommendations_for_imbalance
)
from .threshold_optimization import find_optimal_threshold

__all__ = [
    'FeatureDataset', 'create_patient_level_dataloaders', 'create_test_only_dataloader', 'collate_fn',
    'GatedMILClassifier',
    'calculate_binary_metrics', 'calculate_comprehensive_metrics',
    'plot_confusion_matrix', 'plot_per_class_metrics', 
    'plot_training_metrics', 'plot_roc_curve', 'plot_precision_recall_curve',
    'train_model', 'evaluate_model',
    'set_seed', 'to_json_serializable',
    'setup_logger', 'get_logger', 'log_args', 'log_section',
    'BalancedTrainingConfig', 'calculate_class_weights', 'create_weighted_sampler',
    'get_balanced_loss_function', 'analyze_class_distribution',
    'get_metric_recommendations_for_imbalance',
    'find_optimal_threshold'
] 