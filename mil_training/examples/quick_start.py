#!/usr/bin/env python3
"""
Quick Start Example for  MIL Training

This script demonstrates the most common usage patterns for the MIL training framework.
Run with different configurations to see various training scenarios.
"""

import subprocess
import sys
import os

def run_command(description, command):
    """Run a command and print the description."""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(command)}")
    print(f"{'='*60}")
    
    # Note: In practice, you would uncomment the next line to actually run the command
    # subprocess.run(command, check=True)
    print("📝 Command ready to run (uncomment subprocess.run to execute)")

def main():
    """Demonstrate various training configurations."""
    
    # Base paths (modify these to match your data)
    h5_dir = "/path/to/your/h5_features"
    csv_path = "/path/to/your/labels.csv"
    output_base = "/path/to/your/results"
    
    print("🧬  MIL Training - Quick Start Examples")
    print("=" * 60)
    print("💡 Modify the paths in this script to match your data locations")
    print("🔧 Uncomment subprocess.run() calls to actually execute training")
    
    # Example 1: Basic Binary Classification
    run_command(
        "Basic Binary Classification",
        [
            "python", "train_with_balanced_options.py",
            "--h5_directory", h5_dir,
            "--master_df_path", csv_path,
            "--output_dir", f"{output_base}/basic_binary",
            "--num_epochs", "30",
            "--batch_size", "16"
        ]
    )
    
    # Example 2: Imbalanced Dataset with Balancing
    run_command(
        "Imbalanced Dataset with Class Balancing",
        [
            "python", "train_with_balanced_options.py",
            "--h5_directory", h5_dir,
            "--master_df_path", csv_path,
            "--output_dir", f"{output_base}/balanced_training",
            "--enable_balancing",
            "--balancing_method", "class_weighted",
            "--best_metric", "val_auprc_macro",
            "--early_stopping_patience", "15"
        ]
    )
    
    # Example 3: Regularization for Overfitting
    run_command(
        "Training with Regularization (Anti-Overfitting)",
        [
            "python", "train_with_balanced_options.py",
            "--h5_directory", h5_dir,
            "--master_df_path", csv_path,
            "--output_dir", f"{output_base}/regularized_training",
            "--dropout", "0.2",
            "--attention_dropout", "0.1",
            "--use_layer_norm",
            "--use_skip_connection",
            "--early_stopping_patience", "10",
            "--gradient_clip_value", "1.0"
        ]
    )
    
    # Example 4: Multiclass Classification
    run_command(
        "Multiclass Classification with Advanced Options",
        [
            "python", "train_with_balanced_options.py",
            "--h5_directory", h5_dir,
            "--master_df_path", csv_path,
            "--output_dir", f"{output_base}/multiclass_advanced",
            "--enable_balancing",
            "--balancing_method", "weighted_sampling",
            "--dropout", "0.15",
            "--use_batch_norm",
            "--best_metric", "val_f1_macro",
            "--num_epochs", "50",
            "--batch_size", "24"
        ]
    )
    
    # Example 5: Medical Screening (High Sensitivity)
    run_command(
        "Medical Screening Setup (Prioritizing Sensitivity)",
        [
            "python", "train_with_balanced_options.py",
            "--h5_directory", h5_dir,
            "--master_df_path", csv_path,
            "--output_dir", f"{output_base}/high_sensitivity",
            "--enable_balancing",
            "--weight_method", "effective_samples",
            "--best_metric", "val_sensitivity",
            "--dropout", "0.1",
            "--early_stopping_patience", "20"
        ]
    )
    
    print(f"\n{'='*60}")
    print("✅ All example commands generated!")
    print("📝 Remember to:")
    print("   1. Update the paths to match your data")
    print("   2. Uncomment subprocess.run() calls to execute")
    print("   3. Adjust parameters based on your specific needs")
    print("📚 See README.md for detailed parameter explanations")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
