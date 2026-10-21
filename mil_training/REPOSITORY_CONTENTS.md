# 📁 Repository Contents

This document provides an overview of all files and their purposes in the  MIL Training repository.

## 📂 Root Directory

| File | Purpose |
|------|---------|
| `README.md` | **Main documentation** with comprehensive parameter explanations |
| `requirements.txt` | **Python dependencies** for the project |
| `LICENSE` | MIT license for the project |
| `CHANGELOG.md` | **Version history** and release notes |
| `.gitignore` | Git ignore patterns for Python, data files, and results |

## 🧬 Main Training Scripts

| File | Purpose | When to Use |
|------|---------|-------------|
| `train_with_balanced_options.py` | **Full-featured training script** with all advanced options | ✅ **Primary script** - use for all production training |
| `hyperparameter_search.py` | **Hyperparameter optimization** with random search and k-fold CV | Use for systematic hyperparameter tuning |
| `test_evaluation.py` | **Test set evaluation script** with bootstrap confidence intervals | Use for comprehensive evaluation on held-out test sets |
| `compute_attention.py` | **Attention visualization script** | Use to extract and save attention scores for visualization |
| `postprocess_attention.py` | **Attention post-processing** combining MIL + UNI attention | Use to create granular sub-tile attention maps (14×14 per tile) |

## 📦 Modules Directory (`modules/`)

### Core Components

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| `__init__.py` | **Package initialization** | Exports all public APIs |
| `gated_attn_pool.py` | **MIL model definition** | `GatedMILClassifier` |
| `feature_dataset.py` | **Dataset handling** | `FeatureDataset`, patient-level splitting |
| `training.py` | **Training loop logic** | `train_model`, `evaluate_model` |
| `logger.py` | **Centralized logging** | `setup_logger`, `get_logger`, `log_args`, `log_section` |

### Advanced Features

| File | Purpose | Key Functions |
|------|---------|---------------|
| `balanced_training.py` | **Class balancing utilities** | `BalancedTrainingConfig`, weight calculation |
| `metrics.py` | **Evaluation metrics** | Binary/multiclass metrics calculation |
| `visualization.py` | **Plotting functions** | ROC curves, PR curves, confusion matrices |
| `threshold_optimization.py` | **Threshold optimization** | `find_optimal_threshold` (Youden's J, F1-max, sensitivity targeting) |
| `utils.py` | **General utilities** | `set_seed` and helper functions |


## 📖 Examples Directory (`examples/`)

| File | Purpose | Content |
|------|---------|---------|
| `quick_start.py` | **Ready-to-run examples** | 5 common training scenarios |
| `balanced_training_examples.py` | **Class balancing demonstrations** | Different balancing strategies |
| `metric_selection_examples.py` | **Metric selection guide** | Choosing the right evaluation metric |
| `anti_overfitting_examples.py` | **Regularization examples** | Dropout, normalization, early stopping |
| `batch_norm_analysis.py` | **Normalization comparison** | LayerNorm vs BatchNorm guidance |
| `split_leakage_test.py` | **Patient-level split validation** | Tests to ensure no data leakage between splits |

## 📚 Documentation Directory (`docs/`)

| File | Purpose |
|------|---------|
| `CLASS_BALANCING_GUIDE.md` | **Comprehensive guide to all class balancing parameters** |
| `batch_norm_fix_explanation.md` | **Detailed fix for batch normalization issues** |
| `ATTENTION_POSTPROCESSING.md` | **Complete guide to attention post-processing pipeline** (MIL + UNI attention combination) |

## 🎯 Quick Reference: Which File to Use?

### For Training
```bash
# ✅ Primary choice (recommended)
python train_with_balanced_options.py [options]
```

### For Attention Visualization
```bash
# 1. Compute MIL attention
python compute_attention.py [options]

# 2. Post-process for granular attention (optional)
python postprocess_attention.py [options]
```

### For Understanding
- **Start here**: `README.md`
- **Parameters**: See "Parameters Reference" section in README
- **Examples**: Check `examples/quick_start.py`
- **Troubleshooting**: See "Troubleshooting" section in README
- **Attention post-processing**: See `docs/ATTENTION_POSTPROCESSING.md`

### For Development
- **Model architecture**: `modules/gated_attn_pool.py`
- **Training logic**: `modules/training.py`
- **Class balancing**: `modules/balanced_training.py`
- **Metrics**: `modules/metrics.py`
- **Threshold optimization**: `modules/threshold_optimization.py`

## 🔧 File Dependencies

```
train_with_balanced_options.py
├── modules/__init__.py
├── modules/training.py
│   ├── modules/gated_attn_pool.py
│   ├── modules/feature_dataset.py
│   ├── modules/balanced_training.py
│   ├── modules/metrics.py
│   ├── modules/threshold_optimization.py
│   └── modules/visualization.py
└── modules/utils.py

postprocess_attention.py
├── modules/threshold_optimization.py (for threshold-aware predictions)
└── External: trident (UNI encoder), openslide (WSI loading)
```

## 🚀 Getting Started Workflow

1. **📋 Read**: `README.md` (especially Parameters Reference)
2. **🔧 Install**: `pip install -r requirements.txt`
3. **👀 Explore**: `examples/quick_start.py` for usage patterns
4. **🏃 Run**: `python train_with_balanced_options.py [your_options]`
5. **🔍 Optimize**: Use `hyperparameter_search.py` for systematic hyperparameter tuning
6. **🎯 Threshold**: Use `--operating_point_selection` for binary classification threshold optimization
7. **👁️ Visualize**: Use `compute_attention.py` and `postprocess_attention.py` for attention maps
8. **🐛 Debug**: Check `docs/batch_norm_fix_explanation.md` if issues arise

## 💡 Pro Tips

- **For beginners**: Start with `examples/quick_start.py`
- **For imbalanced data**: Focus on `balanced_training_examples.py`
- **For overfitting**: Check `anti_overfitting_examples.py`
- **For medical applications**: Use `val_auprc_macro` or `val_sensitivity` metrics with `--operating_point_selection sensitivity_95`
- **For small datasets**: Prefer Layer Normalization over Batch Normalization
- **For binary classification**: Use `--operating_point_selection youden` or `f1` for optimal thresholds
- **For granular attention**: Use `postprocess_attention.py` after `compute_attention.py` for 14×14 sub-tile attention maps

---