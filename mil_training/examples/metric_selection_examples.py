"""
Examples showing how to use different metrics for best model selection.

This demonstrates the enhanced metric selection capabilities, including AUPRC
and automatic recommendations based on class distribution.
"""

import os
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from modules import BalancedTrainingConfig, analyze_class_distribution


def example_auprc_for_imbalanced_binary():
    """
    Example: Using AUPRC for imbalanced binary classification
    
    AUPRC is especially good for imbalanced binary classification because:
    - More informative than ROC AUC when positive class is rare
    - Focuses on precision-recall trade-off 
    - Less optimistic than ROC AUC for imbalanced data
    """
    print("=" * 80)
    print("EXAMPLE: AUPRC for Imbalanced Binary Classification")
    print("=" * 80)
    print("Scenario: 900 positive, 100 negative (9:1 imbalance)")
    print("Why AUPRC: More informative than ROC AUC for imbalanced data")
    print("Focus: Precision-Recall trade-off rather than sensitivity-specificity")
    print()
    
    print("Command line usage:")
    print("python train_with_balanced_options.py \\")
    print("    --enable_balancing \\")
    print("    --balancing_method class_weighted_loss \\")
    print("    --best_metric val_auprc_macro")
    print()
    
    # Show what this looks like programmatically
    config = BalancedTrainingConfig(
        enable_balancing=True,
        balancing_method='class_weighted_loss',
        sampling_strategy='natural'
    )
    
    print("Programmatic usage:")
    print("train_model(..., balanced_config=config, best_metric='val_auprc_macro', metric_mode='max')")
    return config


def example_macro_f1_for_multiclass():
    """
    Example: Using Macro F1 for multiclass imbalanced datasets
    
    Macro F1 is good for multiclass because:
    - Gives equal weight to all classes regardless of size
    - Sensitive to performance on minority classes
    - Easy to interpret across different class counts
    """
    print("\n" + "=" * 80)
    print("EXAMPLE: Macro F1 for Multiclass Imbalanced Datasets")
    print("=" * 80)
    print("Scenario: Multiple classes with varying sizes")
    print("Why Macro F1: Equal importance to all classes, sensitive to minority performance")
    print("Alternative: Micro F1 weights by class frequency (not recommended for imbalance)")
    print()
    
    print("Command line usage:")
    print("python train_with_balanced_options.py \\")
    print("    --enable_balancing \\")
    print("    --best_metric val_f1_macro")
    print()
    
    config = BalancedTrainingConfig(
        enable_balancing=True,
        balancing_method='class_weighted_loss'
    )
    
    print("Programmatic usage:")
    print("train_model(..., balanced_config=config, best_metric='val_f1_macro', metric_mode='max')")
    return config


def example_sensitivity_for_medical():
    """
    Example: Using Sensitivity for medical/diagnostic applications
    
    Sensitivity (Recall) is critical when:
    - False negatives are very costly (missing disease)
    - High sensitivity is more important than specificity
    - You want to catch as many positive cases as possible
    """
    print("\n" + "=" * 80)
    print("EXAMPLE: Sensitivity for Medical/Diagnostic Applications")
    print("=" * 80)
    print("Scenario: Medical diagnosis where missing positive cases is critical")
    print("Why Sensitivity: Minimizes false negatives (missed diagnoses)")
    print("Trade-off: May increase false positives")
    print()
    
    print("Command line usage:")
    print("python train_with_balanced_options.py \\")
    print("    --enable_balancing \\")
    print("    --best_metric val_sensitivity")
    print()
    
    config = BalancedTrainingConfig(
        enable_balancing=True,
        balancing_method='class_weighted_loss'
    )
    
    print("Programmatic usage:")
    print("train_model(..., balanced_config=config, best_metric='val_sensitivity', metric_mode='max')")
    return config


def example_mcc_for_extreme_imbalance():
    """
    Example: Using MCC (Matthews Correlation Coefficient) for extreme imbalance
    
    MCC is good for extreme imbalance because:
    - Single metric that considers all confusion matrix elements
    - Ranges from -1 to 1 (like correlation)
    - Less influenced by class distribution than accuracy
    """
    print("\n" + "=" * 80)
    print("EXAMPLE: MCC for Extreme Imbalance")
    print("=" * 80)
    print("Scenario: Extreme imbalance (e.g., 10000:100 ratio)")
    print("Why MCC: Balanced view considering all confusion matrix elements")
    print("Range: -1 (worst) to 1 (perfect), 0 = random")
    print()
    
    print("Command line usage:")
    print("python train_with_balanced_options.py \\")
    print("    --enable_balancing \\")
    print("    --weight_method effective_samples \\")
    print("    --best_metric val_mcc")
    print()
    
    config = BalancedTrainingConfig(
        enable_balancing=True,
        balancing_method='class_weighted_loss',
        weight_method='effective_samples'
    )
    
    print("Programmatic usage:")
    print("train_model(..., balanced_config=config, best_metric='val_mcc', metric_mode='max')")
    return config


def example_auto_mode():
    """
    Example: Using auto mode for metric direction
    
    Auto mode automatically infers whether to minimize or maximize:
    - min: val_loss
    - max: val_acc, val_f1_macro, val_auprc_macro, etc.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE: Auto Mode for Metric Direction")
    print("=" * 80)
    print("Auto mode infers metric direction:")
    print("  Minimize: val_loss")
    print("  Maximize: val_acc, val_f1_macro, val_auprc_macro, val_mcc, etc.")
    print()
    
    print("Command line usage (auto mode is default):")
    print("python train_with_balanced_options.py \\")
    print("    --best_metric val_auprc_macro \\")
    print("    --metric_mode auto")
    print()
    
    print("Or explicitly specify:")
    print("python train_with_balanced_options.py \\")
    print("    --best_metric val_auprc_macro \\")
    print("    --metric_mode max")


def demonstrate_automatic_recommendations():
    """Show how the system automatically recommends metrics based on class distribution."""
    print("\n" + "=" * 80)
    print("AUTOMATIC RECOMMENDATIONS BASED ON CLASS DISTRIBUTION")
    print("=" * 80)
    
    # Simulate different scenarios
    scenarios = [
        {
            'name': 'Balanced Binary (500/500)',
            'labels': [0] * 500 + [1] * 500,
            'class_names': ['Negative', 'Positive']
        },
        {
            'name': 'Moderate Imbalance Binary (900/100)', 
            'labels': [0] * 100 + [1] * 900,
            'class_names': ['Negative', 'Positive']
        },
        {
            'name': 'Extreme Imbalance Binary (9900/100)',
            'labels': [0] * 100 + [1] * 9900,
            'class_names': ['Negative', 'Positive']
        },
        {
            'name': 'Balanced Multiclass (300 each)',
            'labels': [0] * 300 + [1] * 300 + [2] * 300 + [3] * 300,
            'class_names': ['Class A', 'Class B', 'Class C', 'Class D']
        },
        {
            'name': 'Imbalanced Multiclass (600/200/100/100)',
            'labels': [0] * 600 + [1] * 200 + [2] * 100 + [3] * 100,
            'class_names': ['Majority', 'Medium', 'Minority1', 'Minority2']
        }
    ]
    
    for scenario in scenarios:
        print(f"\n--- {scenario['name']} ---")
        distribution = analyze_class_distribution(scenario['labels'], scenario['class_names'])
        print()


def show_all_available_metrics():
    """Show all available metrics for best model selection."""
    print("\n" + "=" * 80)
    print("ALL AVAILABLE METRICS FOR BEST MODEL SELECTION")
    print("=" * 80)
    
    metrics = {
        'val_loss': {
            'description': 'Validation loss (CrossEntropy/BCE)',
            'mode': 'minimize',
            'good_for': 'General purpose, stable training',
            'binary': '✓',
            'multiclass': '✓'
        },
        'val_acc': {
            'description': 'Validation accuracy',
            'mode': 'maximize', 
            'good_for': 'Balanced datasets',
            'binary': '✓',
            'multiclass': '✓'
        },
        'val_f1_macro': {
            'description': 'Macro-averaged F1 score',
            'mode': 'maximize',
            'good_for': 'Imbalanced datasets, equal class importance',
            'binary': '✓',
            'multiclass': '✓'
        },
        'val_f1_micro': {
            'description': 'Micro-averaged F1 score',
            'mode': 'maximize',
            'good_for': 'When larger classes are more important',
            'binary': '✓',
            'multiclass': '✓'
        },
        'val_roc_auc_macro': {
            'description': 'Macro-averaged ROC AUC',
            'mode': 'maximize',
            'good_for': 'Binary classification, ranking quality',
            'binary': '✓',
            'multiclass': '✓'
        },
        'val_auprc_macro': {
            'description': 'Macro-averaged AUPRC (Average Precision)',
            'mode': 'maximize',
            'good_for': 'Imbalanced datasets, precision-recall trade-off',
            'binary': '✓',
            'multiclass': '✓'
        },
        'val_mcc': {
            'description': 'Matthews Correlation Coefficient',
            'mode': 'maximize',
            'good_for': 'Extreme imbalance, balanced assessment',
            'binary': '✓',
            'multiclass': '✓'
        },
        'val_sensitivity': {
            'description': 'Sensitivity (Recall, True Positive Rate)',
            'mode': 'maximize',
            'good_for': 'When missing positives is costly',
            'binary': '✓',
            'multiclass': '×'
        },
        'val_specificity': {
            'description': 'Specificity (True Negative Rate)',
            'mode': 'maximize',
            'good_for': 'When false positives are costly',
            'binary': '✓',
            'multiclass': '×'
        }
    }
    
    print(f"{'Metric':<20} {'Description':<40} {'Mode':<10} {'Binary':<8} {'Multi':<8}")
    print("-" * 90)
    
    for metric, info in metrics.items():
        print(f"{metric:<20} {info['description']:<40} {info['mode']:<10} {info['binary']:<8} {info['multiclass']:<8}")
    
    print(f"\n💡 QUICK RECOMMENDATIONS:")
    print(f"  🎯 Imbalanced binary: val_auprc_macro")
    print(f"  🎯 Imbalanced multiclass: val_f1_macro") 
    print(f"  🎯 Medical diagnosis: val_sensitivity")
    print(f"  🎯 Extreme imbalance: val_mcc")
    print(f"  🎯 Balanced datasets: val_acc or val_f1_macro")


def main():
    """Run all examples demonstrating metric selection."""
    print("METRIC SELECTION FOR BEST MODEL EXAMPLES")
    print("Enhanced training with configurable metric selection and AUPRC support")
    
    # Run examples
    example_auprc_for_imbalanced_binary()
    example_macro_f1_for_multiclass()
    example_sensitivity_for_medical()
    example_mcc_for_extreme_imbalance()
    example_auto_mode()
    
    # Show automatic recommendations
    demonstrate_automatic_recommendations()
    
    # Show all available metrics
    show_all_available_metrics()
    
    print(f"\n" + "=" * 80)
    print("KEY TAKEAWAYS")
    print("=" * 80)
    print("1. ✅ AUPRC is now fully supported and recommended for imbalanced data")
    print("2. ✅ Automatic recommendations based on your dataset's class distribution")
    print("3. ✅ Multiclass-specific adjustments (macro vs micro averaging)")
    print("4. ✅ Medical/diagnostic focused metrics (sensitivity, specificity)")
    print("5. ✅ Auto mode infers whether to minimize or maximize metrics")
    print("6. ✅ All metrics are tracked during training for comprehensive analysis")
    
    print(f"\n🚀 USAGE:")
    print(f"python train_with_balanced_options.py --best_metric val_auprc_macro")
    print(f"python train_with_balanced_options.py --best_metric val_f1_macro") 
    print(f"python train_with_balanced_options.py --best_metric val_sensitivity")


if __name__ == "__main__":
    main()
