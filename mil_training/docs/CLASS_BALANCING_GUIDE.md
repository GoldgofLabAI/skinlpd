# 🔄 Class Balancing Configuration Guide

This guide explains all class balancing parameters and their interactions in the  MIL Training framework.

## 📋 Table of Contents

- [🎯 Quick Reference](#quick-reference)
- [⚖️ Balancing Methods](#balancing-methods)
- [📊 Sampling Strategies](#sampling-strategies)
- [🔢 Weight Calculation Methods](#weight-calculation-methods)
- [🔧 Weight Smoothing](#weight-smoothing)
- [🤝 Parameter Interactions](#parameter-interactions)
- [📖 Complete Examples](#complete-examples)
- [💡 Choosing the Right Strategy](#choosing-the-right-strategy)

## 🎯 Quick Reference

| Parameter | Options | Default | Purpose |
|-----------|---------|---------|---------|
| `--balancing_method` | `class_weighted`, `weighted_sampling` | `class_weighted` | **Where** to apply balancing |
| `--sampling_strategy` | `balanced`, `custom` | `balanced` | **How much** to resample |
| `--weight_method` | `inverse_frequency`, `balanced`, `effective_samples` | `inverse_frequency` | **How** to calculate weights |
| `--smooth_weights` | flag | disabled | **Whether** to smooth extreme weights |

## ⚖️ Balancing Methods

### `--balancing_method class_weighted` ✅ **Recommended**

**What it does**: Keeps the original data distribution but adjusts the **loss function** to give more weight to minority classes.

**How it works**:
```python
# Training batches maintain natural distribution (e.g., 90% positive, 10% negative)
# But loss function weights minority class errors more heavily

loss = CrossEntropyLoss(weight=class_weights)
# class_weights = [weight_negative, weight_positive]
# Example: [0.56, 5.0] for 9:1 imbalance
```

**Advantages**:
- ✅ **No duplicate samples** → no overfitting risk
- ✅ **Stable training** → reflects real data distribution
- ✅ **Memory efficient** → no additional sampling overhead
- ✅ **Fast convergence** → proven in medical literature

**Disadvantages**:
- 🟡 Minority class may still be underexposed in very extreme imbalance (>100:1)

**Best for**: Most imbalanced datasets (2:1 to 50:1 ratio)

### `--balancing_method weighted_sampling`

**What it does**: **Resamples** the training data to create more balanced batches.

**How it works**:
```python
# Original dataset: 900 positive, 100 negative
# Weighted sampling creates batches like: 16 positive, 16 negative
# Achieved by resampling minority class multiple times per epoch

sampler = WeightedRandomSampler(sample_weights, num_samples=len(dataset))
```

**Advantages**:
- ✅ **Guaranteed exposure** → minority class in every batch
- ✅ **Strong signal** → balanced gradients every update
- ✅ **Effective for extreme imbalance** → works even at 1000:1 ratios

**Disadvantages**:
- ⚠️ **Overfitting risk** → minority samples seen multiple times
- ⚠️ **Slower training** → more total samples per epoch
- ⚠️ **Memory overhead** → requires storing sample weights

**Best for**: Severe imbalance (>50:1) or when minority class is very small (<50 samples)

## 📊 Sampling Strategies

*Note: Only applies when `--balancing_method weighted_sampling`*

### `--sampling_strategy balanced` ✅ **Default**

**What it does**: Attempts to create **perfectly balanced** batches where each class appears equally.

**How it works**:
```python
# For binary classification: 50% class 0, 50% class 1
# For 3-class: 33.3% each class
# For 4-class: 25% each class

target_ratio = 1.0 / num_classes
```

**Example**:
```bash
# Original: 900 positive, 100 negative
# Balanced batches: ~16 positive, ~16 negative (batch_size=32)
```

### `--sampling_strategy custom`

**What it does**: Uses **custom sampling ratios** (currently implements the same as balanced, but extensible).

**Future extensions**: Could support custom ratios like 70:30 instead of 50:50.

## 🔢 Weight Calculation Methods

### `--weight_method inverse_frequency` ✅ **Default**

**Formula**:
```python
weight[i] = total_samples / class_count[i]
```

**Example** (1000 samples: 900 class A, 100 class B):
```python
weight_A = 1000 / 900 = 1.11
weight_B = 1000 / 100 = 10.0
# Ratio: 1:9 → minority class weighted 9x more
```

**Characteristics**:
- ✅ **Simple and intuitive** → directly proportional to rarity
- ✅ **Strong correction** → high weights for rare classes
- ⚠️ **Can be extreme** → very high weights for very rare classes

**Best for**: Moderate imbalance (2:1 to 20:1)

### `--weight_method balanced`

**Formula** (sklearn-style):
```python
weight[i] = total_samples / (num_classes * class_count[i])
```

**Example** (same 1000 samples: 900 class A, 100 class B):
```python
weight_A = 1000 / (2 * 900) = 0.56
weight_B = 1000 / (2 * 100) = 5.0
# Ratio: 1:9 → but normalized by number of classes
```

**Characteristics**:
- ✅ **Normalized approach** → weights sum to reasonable values
- ✅ **Multi-class friendly** → scales well with more classes
- 🟡 **Gentler correction** → less extreme than inverse_frequency

**Best for**: Multi-class problems or when inverse_frequency gives extreme weights

### `--weight_method effective_samples`

**Formula** (for very imbalanced data):
```python
beta = 0.9999  # smoothing parameter
effective_count[i] = (1 - beta) / (1 - beta^class_count[i])
weight[i] = 1.0 / effective_count[i]
```

**Example** (same 1000 samples):
```python
# For class A (900 samples): effective_count ≈ 900 (no change)
# For class B (100 samples): effective_count ≈ 99.9 (slight smoothing)
# Results in smoother, less extreme weights
```

**Characteristics**:
- ✅ **Handles extreme imbalance** → designed for 100:1+ ratios
- ✅ **Smooth transitions** → avoids sudden weight jumps
- ✅ **Theoretically grounded** → based on effective number of samples

**Best for**: Extreme imbalance (>100:1) or very small minority classes

## 🔧 Weight Smoothing

### `--smooth_weights` 🔧 **Optional Flag**

**What it does**: Applies **additional smoothing** to calculated weights to prevent extreme values.

**How it works**:
```python
# Before smoothing: [1.11, 10.0]
# After smoothing: [1.5, 6.0]  (example - reduces extremes)

# Current implementation uses min-max normalization
# to keep weights in a reasonable range
```

**When to use**:
- ✅ **Very extreme imbalance** → when raw weights are >20:1
- ✅ **Training instability** → if loss oscillates wildly
- ✅ **Conservative approach** → when you want gentler balancing

**Trade-offs**:
- 🟡 **Less aggressive** → may not fully correct severe imbalance
- ✅ **More stable** → reduces risk of training instability

## 🤝 Parameter Interactions

### **Combination Matrix**

| Method | Sampling | Weight Method | Smooth | Result |
|--------|----------|---------------|--------|--------|
| `class_weighted` | *N/A* | `inverse_frequency` | ❌ | **Strong correction, natural sampling** |
| `class_weighted` | *N/A* | `balanced` | ❌ | **Moderate correction, multi-class friendly** |
| `class_weighted` | *N/A* | `effective_samples` | ❌ | **Smooth correction for extreme imbalance** |
| `class_weighted` | *N/A* | `inverse_frequency` | ✅ | **Tempered correction, stable training** |
| `weighted_sampling` | `balanced` | `inverse_frequency` | ❌ | **Balanced batches + loss weighting** |
| `weighted_sampling` | `balanced` | `balanced` | ✅ | **Conservative balanced approach** |

### **Priority Rules**

1. **`balancing_method`** determines the primary strategy
2. **`weight_method`** only affects weight calculation (always used)
3. **`sampling_strategy`** only applies to `weighted_sampling`
4. **`smooth_weights`** post-processes any calculated weights

## 📖 Complete Examples

### **Example 1: Medical Screening (Conservative)**
```bash
# Goal: High sensitivity, stable training
python train_with_balanced_options.py \
    --enable_balancing \
    --balancing_method class_weighted \
    --weight_method balanced \
    --smooth_weights \
    --best_metric val_sensitivity
```

**Result**: Gentle class weighting with smoothed extremes for stable training.

### **Example 2: Severe Imbalance (Aggressive)**
```bash
# Goal: Handle 100:1 imbalance effectively
python train_with_balanced_options.py \
    --enable_balancing \
    --balancing_method weighted_sampling \
    --sampling_strategy balanced \
    --weight_method effective_samples \
    --best_metric val_auprc_macro
```

**Result**: Balanced batches + smooth weights designed for extreme imbalance.

### **Example 3: Multiclass Tissue Typing**
```bash
# Goal: Fair performance across all tissue types
python train_with_balanced_options.py \
    --enable_balancing \
    --balancing_method class_weighted \
    --weight_method balanced \
    --best_metric val_f1_macro
```

**Result**: Multi-class normalized weighting for fair classification.

### **Example 4: Small Dataset (Cautious)**
```bash
# Goal: Avoid overfitting with limited data
python train_with_balanced_options.py \
    --enable_balancing \
    --balancing_method class_weighted \
    --weight_method inverse_frequency \
    --smooth_weights \
    --dropout 0.2 \
    --early_stopping_patience 10
```

**Result**: Class weighting with overfitting prevention.

## 💡 Choosing the Right Strategy

### **By Imbalance Ratio**

| Imbalance Ratio | Recommended Configuration |
|-----------------|---------------------------|
| **2:1 to 5:1** | `class_weighted` + `inverse_frequency` |
| **5:1 to 20:1** | `class_weighted` + `balanced` |
| **20:1 to 50:1** | `class_weighted` + `effective_samples` |
| **50:1 to 100:1** | `weighted_sampling` + `balanced` + `effective_samples` |
| **>100:1** | `weighted_sampling` + `balanced` + `effective_samples` + `smooth_weights` |

### **By Dataset Size**

| Dataset Size | Recommendation |
|--------------|----------------|
| **Large (>10K samples)** | Any method, prefer `class_weighted` |
| **Medium (1K-10K)** | `class_weighted` preferred, avoid extreme weights |
| **Small (<1K)** | `class_weighted` + `smooth_weights` + regularization |
| **Very small (<100)** | `class_weighted` + `smooth_weights` + heavy regularization |

### **By Application**

| Application | Priority | Recommended Setup |
|-------------|----------|-------------------|
| **Medical Screening** | High Sensitivity | `effective_samples` + `val_sensitivity` |
| **Quality Control** | High Specificity | `balanced` + `val_specificity` |
| **Research** | Balanced Performance | `balanced` + `val_f1_macro` |
| **Production** | Stable Training | `class_weighted` + `smooth_weights` |

### **Decision Tree**

```
Is your imbalance ratio > 50:1?
├── Yes → Use weighted_sampling + effective_samples
└── No → Use class_weighted
    ├── Multiclass? → Use balanced weight method
    ├── Extreme weights (>20:1)? → Add smooth_weights
    └── Small dataset (<1K)? → Add smooth_weights + regularization
```

## 🔍 Debugging Class Balancing

### **Check Class Distribution**
The framework automatically prints class analysis:
```
Class Distribution Analysis:
  Class 0 (negative): 100 samples (10.0%)
  Class 1 (positive): 900 samples (90.0%)
  Imbalance ratio: 9.0:1 (majority:minority)
  
Calculated weights: [5.0, 0.56]
```

### **Monitor Training Metrics**
- **Sensitivity**: Recall for positive class
- **Specificity**: Recall for negative class  
- **F1-macro**: Balanced F1 across classes
- **AUPRC**: Area under precision-recall curve

### **Signs of Good Balancing**
- ✅ Both sensitivity and specificity > 0.7
- ✅ F1-macro close to individual class F1 scores
- ✅ AUPRC significantly > class prevalence
- ✅ Stable training curves

### **Signs of Poor Balancing**
- ⚠️ One class with very low recall (<0.3)
- ⚠️ Large gap between training and validation metrics
- ⚠️ Oscillating loss curves
- ⚠️ AUPRC barely above random baseline

---

## 📚 References

- **Class-weighted loss**: Widely used in medical AI literature
- **Weighted sampling**: Common in computer vision for long-tail distributions
- **Effective samples**: From "Class-Balanced Loss Based on Effective Number of Samples" (CVPR 2019)
- **Balanced weights**: Standard in scikit-learn and PyTorch

---

**💡 Pro Tip**: Start with `class_weighted` + `balanced` + `smooth_weights` for most applications. Only move to `weighted_sampling` for very severe imbalance or very small minority classes.
