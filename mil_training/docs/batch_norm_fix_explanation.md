# Batch Normalization Fix for Single-Sample Batches

## Problem Encountered

```
ValueError: Expected more than 1 value per channel when training, got input size torch.Size([1, 1024])
```

## Root Cause

**Batch Normalization requires batch size > 1** during training because:
- It calculates mean and variance across the batch dimension
- Single sample batches (size 1) have zero variance, causing division by zero
- This typically happens in the **last batch of an epoch** when `drop_last=False`

## Solutions Implemented

### 1. **Graceful Fallback in Model** ✅
```python
# In GatedMILClassifier
if self.use_batch_norm:
    # Only apply batch norm if batch size > 1
    if pooled_features.size(0) > 1:
        pooled_features = self.batch_norm_pooled(pooled_features)
    # For batch size = 1, skip batch norm (fallback gracefully)
```

### 2. **Drop Last Batch in DataLoaders** ✅
```python
# In training.py
train_loader = DataLoader(..., drop_last=True)

# In feature_dataset.py  
train_dataloader = DataLoader(..., drop_last=True)
```

### 3. **User Warnings** ✅
```
✅ BatchNorm can provide strong regularization for overfitting
📋 Note: Using drop_last=True to prevent single-sample batches
💡 If you have very small datasets, consider LayerNorm instead
```

## When This Issue Occurs

1. **Small datasets** where `dataset_size % batch_size == 1`
2. **Patient-level splitting** that results in uneven batch distributions
3. **Last batch of epoch** with default `drop_last=False`

## Alternative Solutions

If you prefer **not to drop the last batch**, use **Layer Normalization** instead:

```bash
# Instead of batch norm
python train_with_balanced_options.py --use_batch_norm

# Use layer norm (no batch size dependency)
python train_with_balanced_options.py --use_layer_norm
```

## Trade-offs

| Solution | Pros | Cons |
|----------|------|------|
| **Drop Last + BatchNorm** | Strong regularization, no errors | Loses 1-15 samples per epoch |
| **LayerNorm** | Uses all samples, stable | Weaker regularization effect |
| **Graceful Fallback** | Best of both worlds | Inconsistent normalization |

## Recommendation

✅ **Current implementation (Drop Last + Graceful Fallback)** is the best approach:
- Prevents errors completely
- Maximizes BatchNorm usage
- Minimal data loss (typically <1% of samples)
- Provides user warnings for transparency

## Usage

No changes needed in your command line usage! The fix is automatic:

```bash
python train_with_balanced_options.py \
    --use_batch_norm \
    --dropout 0.1 \
    --early_stopping_patience 5 \
    --best_metric val_auprc_macro \
    --enable_balancing
```

The system now handles single-sample batches gracefully. ✨
