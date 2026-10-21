# Changelog

All notable changes to the  MIL Training Framework will be documented in this file.

## [Unreleased]

### ✨ Features Added
- **Case-Level Evaluation**: Aggregate slide predictions to case-part level in `test_evaluation.py` and training operating-point selection
  - `--case_level` with aggregation via `--case_level_method` (`mean` or `max`)
  - Configurable `--case_id_column` and `--part_column` for grouping slides by (case_id, part)
  - Metrics, plots, and exports reflect case-part units when enabled
- **Held-Out Test Split**: New `--test_split` in `train_with_balanced_options.py`
  - Patient-level reservation of a held-out test set during training
  - Saves `test_split_patient_level.csv` mirroring the master CSV schema for external evaluation
- **Recompute Operating Point Without Retraining**: New `--recompute_operating_point` flag
  - Reloads existing `best_model.pt`, recomputes threshold on the validation set, and updates the checkpoint
  - Supports slide-level or case-level aggregation for threshold search
- **Eval-Only Mode**: New `--eval_only` flag to load `best_model.pt` and run validation evaluation without retraining
- **Raw Slide Probability Export**: Case-level evaluation now saves `raw_slide_probabilities.csv`
  - Preserves per-slide probabilities alongside aggregated case-part metrics for downstream analysis
- **Threshold Optimization**: New `modules/threshold_optimization.py` for finding optimal classification thresholds
  - Support for Youden's J statistic (maximize sensitivity + specificity)
  - F1-score maximization
  - Fixed sensitivity targeting (e.g., 95% sensitivity for medical screening)
  - Automatic threshold saving in model checkpoints
  - Integration with training, evaluation, and inference scripts
- **Attention Post-Processing Pipeline**: New `postprocess_attention.py` script for granular attention visualization
  - Combines MIL tile-level attention with UNI encoder token-level attention (14×14 per tile)
  - Produces sub-tile granularity attention maps for whole slide images
  - Automatic handling of 20x vs 40x slide magnifications
  - Multi-GPU support for efficient processing
  - Thumbnail visualization with attention heatmap overlays
  - Comprehensive documentation in `docs/ATTENTION_POSTPROCESSING.md`
- **Operating Point Selection**: New `--operating_point_selection` argument in training script
  - Choose optimal threshold method during training: `default`, `youden`, `f1`, or `sensitivity_95`
  - Optimal threshold automatically saved to model checkpoint
  - Used automatically in `compute_attention.py` and `test_evaluation.py` for consistent predictions
- **Hyperparameter Search Script**: New `hyperparameter_search.py` for systematic hyperparameter optimization
  - Random search with configurable search spaces
  - Support for both train/validation split and k-fold cross-validation
  - Automatic retraining on full dataset with best hyperparameters
  - Flexible search spaces: continuous ranges, discrete options, and boolean flags
  - Single value support: Can fix parameters by providing a single value (e.g., `--attention_dropout_range 0.0`)
- **Centralized Logging System**: New `modules/logger.py` with unified logging utilities
  - File and console logging with configurable levels
  - Structured logging with sections and argument logging
  - Optional detailed training logs via `--save_training_logs_too` flag
- **Conditional Model Saving**: Granular control over model checkpoint saving
  - `--save_final_model`: Save final_model.pt at end of training (default: disabled)
  - `--no_save_best_model`: Disable saving best_model.pt during training (default: enabled)
  - Useful for saving disk space during hyperparameter searches

### 🔧 Improvements
- **Operating Point Metadata in Evaluation Results**: `test_evaluation.py` now records `optimal_threshold` and `operating_point_method` in saved evaluation config JSON
- **JSON Serialization Utility**: Added `to_json_serializable` in `modules/utils.py` for robust saving of numpy/torch types in results JSON
- **PyTorch Checkpoint Compatibility**: Model loading uses `weights_only=False` for broader checkpoint compatibility; package versions updated in `requirements.txt`
- **Threshold-Aware Inference**: `compute_attention.py` and `test_evaluation.py` now use optimal thresholds when available
  - Automatic detection of threshold in model checkpoint
  - Threshold-based predictions for binary classification (instead of default 0.5)
  - Backward compatible: falls back to argmax if no threshold found
- **Range Parameter Flexibility**: Range parameters now accept both single values and ranges
  - Single value: `--attention_dropout_range 0.0` (fixed value, no sampling)
  - Range: `--attention_dropout_range 0.0,0.5` (sample from range)
- **Logging Integration**: All main scripts now use centralized logging system
  - `train_with_balanced_options.py`: Full logging support with optional detailed training logs
  - `hyperparameter_search.py`: Comprehensive logging of search process
  - `modules/training.py`: Replaced print statements with logger calls
  - `postprocess_attention.py`: Comprehensive logging with multi-GPU support

### 🐛 Bug Fixes
- **Precision-Recall Curve Labels**: Fixed label formatting in precision-recall curve plots
- **Directory Creation**: Fixed unnecessary directory creation when model saving is disabled
- **Metric Extraction**: Improved handling of per-epoch metric lists vs single values
- **Boolean Options Parsing**: Enhanced robustness for parsing boolean search options
- **Magnification Handling**: Improved handling of 20x vs 40x slides in attention post-processing
  - Proper patch size detection from H5 attributes
  - Correct visualization placement regardless of native magnification

## [1.0.0] - 2025-09-04

### 🎉 Initial Release

#### ✨ Features Added
- **Modular Architecture**: Refactored monolithic training script into clean, reusable modules
- **Advanced Class Balancing**: 
  - Natural sampling + class-weighted loss
  - Weighted random sampling  
  - Multiple weight calculation methods (inverse frequency, balanced, effective samples)
- **Comprehensive Regularization**:
  - Configurable dropout (general + attention-specific)
  - Layer Normalization and Batch Normalization support
  - Skip connections for improved gradient flow
  - Early stopping with patience
  - Gradient clipping
- **Flexible Metric Selection**: Choose from 10+ metrics for best model selection
- **Enhanced Visualization**:
  - ROC curves with macro/micro averaging
  - Precision-Recall curves for binary classification
  - Comprehensive confusion matrices
  - Training metric plots
- **Robust Error Handling**: Graceful fallback for batch normalization edge cases

#### 🏗️ Architecture Components
- `GatedMILClassifier`: Attention-based MIL model with configurable regularization
- `FeatureDataset`: Efficient H5 feature loading with patient-level splitting
- `BalancedTrainingConfig`: Unified class balancing configuration
- Comprehensive metrics calculation (binary + multiclass)
- Advanced visualization suite

#### 📊 Supported Metrics
- Accuracy, Precision, Recall, F1-score (macro/micro)
- ROC AUC (macro/micro)
- Average Precision/AUPRC (macro/micro)
- Sensitivity, Specificity
- Matthews Correlation Coefficient (MCC)

#### 🛡️ Regularization Techniques
- **Dropout**: Input features, attention scores, classifier layers
- **Normalization**: Layer normalization (stable) or Batch normalization (strong)
- **Architecture**: Configurable latent dimensions, skip connections
- **Training**: Early stopping, gradient clipping, weight initialization

#### ⚖️ Class Balancing Methods
- **Class-weighted loss**: Adjust loss function weights based on class frequency
- **Weighted sampling**: Resample training data to balance batches
- **Weight calculation**: Inverse frequency, balanced (sklearn-style), effective samples
- **Automatic detection**: Multiclass-compatible with automatic adjustments

#### 🔧 Configuration Options
- **40+ command-line arguments** for comprehensive control
- **Automatic metric mode detection** (minimize/maximize)
- **Patient-level data splitting** to prevent data leakage
- **Flexible output directory structure**

### 🐛 Bug Fixes
- **Batch Normalization Error**: Fixed `ValueError` for single-sample batches with graceful fallback
- **Tensor Shape Compatibility**: Ensured CrossEntropyLoss compatibility for binary classification
- **Memory Management**: Added `drop_last=True` to prevent batch size issues

### 📚 Documentation
- **Comprehensive README**: Detailed parameter explanations and usage examples
- **Quick Start Guide**: Ready-to-run examples for common scenarios
- **Troubleshooting Guide**: Solutions for common issues and performance optimization
- **Example Scripts**: Demonstration of balancing, regularization, and metric selection

### 🎯 Use Cases Supported
- **Binary Classification**: Medical diagnosis, screening applications
- **Multiclass Classification**: Tissue subtyping, multi-organ classification  
- **Imbalanced Datasets**: Rare disease detection, screening scenarios
- **Overfitting Control**: Small datasets, complex models
- **Medical Applications**: High sensitivity/specificity requirements

### 🔄 Usage Guide
1. Install dependencies: `pip install -r requirements.txt`
2. Update import statements to use the new modular structure
3. Use `train_with_balanced_options.py` for the full feature set
4. See `examples/` directory for migration examples

---

**📝 Note**: This changelog follows [Keep a Changelog](https://keepachangelog.com/) format.
