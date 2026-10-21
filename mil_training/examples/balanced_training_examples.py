"""
Examples demonstrating how to use the balanced training functionality.

This script shows different configurations for handling class imbalance based on the 
ChatGPT recommendations you received.
"""

import os
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from modules import BalancedTrainingConfig, set_seed


def example_natural_sampling_with_class_weights():
    """
    Example 1: Natural sampling + class-weighted loss (Recommended baseline)
    
    This is the approach recommended in your ChatGPT conversation as a simple and strong baseline.
    It keeps the natural data distribution but weights the loss function to account for imbalance.
    """
    print("=" * 80)
    print("EXAMPLE 1: Natural Sampling + Class-Weighted Loss")
    print("=" * 80)
    print("Best for: Most imbalanced datasets, stable training")
    print("Benefits: No duplicate samples, reflects real priors, stable gradients")
    print("Use when: You have moderate to high imbalance (like your 900/100 example)")
    print()
    
    # Configuration for natural sampling with class-weighted loss
    config = BalancedTrainingConfig(
        enable_balancing=True,
        balancing_method='class_weighted_loss',  # Use weighted loss only
        sampling_strategy='natural',             # Keep natural distribution
        weight_method='balanced',                # sklearn-style balanced weights
        smooth_weights=0.0                       # No smoothing
    )
    
    config.print_config()
    print("\nExample command line usage:")
    print("python train_with_balanced_options.py \\")
    print("    --enable_balancing \\")
    print("    --balancing_method class_weighted_loss \\")
    print("    --sampling_strategy natural \\")
    print("    --weight_method balanced")
    return config


def example_weighted_sampling():
    """
    Example 2: Weighted sampling for balanced batches
    
    This ensures each batch has roughly equal representation from each class.
    Good for ensuring minority class exposure but risks overfitting.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Weighted Sampling (Balanced Batches)")
    print("=" * 80)
    print("Best for: Extreme imbalance where minority class rarely appears")
    print("Benefits: Guarantees minority exposure each batch")
    print("Risks: Duplicate minority samples → overfitting")
    print("Mitigation: Use with data augmentation, dropout, weight decay")
    print()
    
    config = BalancedTrainingConfig(
        enable_balancing=True,
        balancing_method='weighted_sampling',    # Use sampling only
        sampling_strategy='balanced',            # Balanced sampling
        weight_method='balanced',
        smooth_weights=0.0
    )
    
    config.print_config()
    print("\nExample command line usage:")
    print("python train_with_balanced_options.py \\")
    print("    --enable_balancing \\")
    print("    --balancing_method weighted_sampling \\")
    print("    --sampling_strategy balanced")
    return config


def example_both_techniques():
    """
    Example 3: Both weighted sampling and class-weighted loss
    
    Combines both techniques for maximum balance, but may be overkill for most cases.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Both Weighted Sampling + Class-Weighted Loss")
    print("=" * 80)
    print("Best for: Extreme imbalance where other methods fail")
    print("Benefits: Maximum balance correction")
    print("Risks: May overcorrect, leading to poor generalization")
    print("Use carefully: Monitor validation performance closely")
    print()
    
    config = BalancedTrainingConfig(
        enable_balancing=True,
        balancing_method='both',                 # Use both techniques
        sampling_strategy='balanced',            # Balanced sampling
        weight_method='balanced',
        smooth_weights=0.0
    )
    
    config.print_config()
    print("\nExample command line usage:")
    print("python train_with_balanced_options.py \\")
    print("    --enable_balancing \\")
    print("    --balancing_method both \\")
    print("    --sampling_strategy balanced")
    return config


def example_binary_classification_specific():
    """
    Example 4: Binary classification with specific positive weight
    
    For binary classification, you can specify the exact positive class weight.
    Following the 900/100 example from ChatGPT: pos_weight = n_neg/n_pos = 100/900 ≈ 0.111
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Binary Classification with Specific Positive Weight")
    print("=" * 80)
    print("Best for: Binary classification with known class counts")
    print("Example: 900 positive, 100 negative → pos_weight = 100/900 = 0.111")
    print("This tells the model: 'treat negative class mistakes as ~9x more important'")
    print()
    
    config = BalancedTrainingConfig(
        enable_balancing=True,
        balancing_method='class_weighted_loss',
        sampling_strategy='natural',
        pos_weight=0.111,                        # Specific weight for binary case
        weight_method='balanced'
    )
    
    config.print_config()
    print("\nExample command line usage:")
    print("python train_with_balanced_options.py \\")
    print("    --enable_balancing \\")
    print("    --balancing_method class_weighted_loss \\")
    print("    --pos_weight 0.111")
    return config


def example_effective_samples_weighting():
    """
    Example 5: Effective samples weighting for extreme imbalance
    
    Uses effective number of samples approach, good for very extreme imbalance.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Effective Samples Weighting (Extreme Imbalance)")
    print("=" * 80)
    print("Best for: Very extreme imbalance (e.g., 10000:10 ratio)")
    print("Benefits: Better handling of very rare classes")
    print("Based on: 'Class-Balanced Loss Based on Effective Number of Samples'")
    print()
    
    config = BalancedTrainingConfig(
        enable_balancing=True,
        balancing_method='class_weighted_loss',
        sampling_strategy='natural',
        weight_method='effective_samples',       # Different weighting method
        smooth_weights=0.0
    )
    
    config.print_config()
    print("\nExample command line usage:")
    print("python train_with_balanced_options.py \\")
    print("    --enable_balancing \\")
    print("    --weight_method effective_samples")
    return config


def example_no_balancing():
    """
    Example 6: No balancing (baseline)
    
    Standard training without any balancing techniques.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 6: No Balancing (Baseline)")
    print("=" * 80)
    print("Best for: Relatively balanced datasets or when you want baseline comparison")
    print("Benefits: Simple, fast, no risk of overcorrection")
    print("Use when: Class imbalance ratio < 3:1")
    print()
    
    config = BalancedTrainingConfig(
        enable_balancing=False               # Disabled
    )
    
    config.print_config()
    print("\nExample command line usage:")
    print("python train_with_balanced_options.py")
    print("# (no balancing flags needed)")
    return config


def get_recommendation_for_imbalance_ratio(imbalance_ratio):
    """
    Provide recommendations based on imbalance ratio.
    
    Args:
        imbalance_ratio (float): Ratio of majority to minority class
    
    Returns:
        str: Recommendation
    """
    if imbalance_ratio < 2:
        return "Balanced dataset - no balancing needed"
    elif imbalance_ratio < 5:
        return "Mild imbalance - try natural sampling + class weights first"
    elif imbalance_ratio < 20:
        return "Moderate imbalance - natural sampling + class weights recommended"
    elif imbalance_ratio < 100:
        return "High imbalance - consider weighted sampling or effective samples method"
    else:
        return "Extreme imbalance - use effective samples + weighted sampling"


def main():
    """Run all examples to demonstrate different configurations."""
    print("BALANCED TRAINING CONFIGURATION EXAMPLES")
    print("Based on ChatGPT recommendations for handling class imbalance")
    print("\nThese examples show how to configure balanced training for different scenarios.")
    
    # Run all examples
    configs = [
        example_natural_sampling_with_class_weights(),
        example_weighted_sampling(),
        example_both_techniques(),
        example_binary_classification_specific(),
        example_effective_samples_weighting(),
        example_no_balancing()
    ]
    
    # Show recommendations for different imbalance ratios
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS BY IMBALANCE RATIO")
    print("=" * 80)
    
    test_ratios = [1.5, 3, 9, 50, 500]  # Including your 900/100 = 9 example
    for ratio in test_ratios:
        recommendation = get_recommendation_for_imbalance_ratio(ratio)
        print(f"Ratio {ratio}:1 → {recommendation}")
    
    print(f"\nYour example (900/100 = 9:1) → {get_recommendation_for_imbalance_ratio(9)}")
    
    print("\n" + "=" * 80)
    print("KEY TAKEAWAYS FROM CHATGPT RECOMMENDATIONS")
    print("=" * 80)
    print("1. Start with natural sampling + class-weighted loss (Example 1)")
    print("2. Keep validation/test sets at natural distribution (never balance them)")
    print("3. Use AUPRC metric (more informative than AUROC for imbalanced data)")
    print("4. Tune decision threshold on validation set based on your cost function")
    print("5. Track per-class precision/recall, not just overall accuracy")
    print("6. For extreme imbalance, consider weighted sampling with strong regularization")
    print("7. Monitor for overfitting when using balanced sampling")
    print("8. Consider focal loss for hard example mining (not implemented yet)")
    
    print(f"\nTo run training with balanced options:")
    print(f"python train_with_balanced_options.py --help")


if __name__ == "__main__":
    main()
