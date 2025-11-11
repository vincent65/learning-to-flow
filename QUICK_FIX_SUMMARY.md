# Mode Collapse Fixes - Quick Summary

## The Problem

After 1 epoch with normalization fixes, your model:
- ✅ Moves smoothly (movement=0.24, step_distance=0.024)
- ❌ **All embeddings collapse to ONE point [21,21]** (mode collapse)
- ❌ Silhouette by target = 1.0 (should be 0.35-0.55)

**Why?** Model learned "move smoothly to one safe location" instead of "move to different locations based on attributes"

---

## Three Critical Fixes Applied

### 1. Reduce Identity Loss (configs/fclf_config.yaml)
```yaml
# CHANGED
lambda_identity: 1.0 → 0.05  # Reduced by 20x
```
**Why**: Identity loss was TOO STRONG, preventing attribute-specific movement

### 2. Always Apply Attribute Augmentation (src/training/train_fclf.py)
```python
# REMOVED: if torch.rand(1).item() > 0.5:
# NOW: ALWAYS flip 1-2 attributes per sample
```
**Why**: Model needs consistent transfer training, not 50% identity preservation

### 3. Prototype-Based Contrastive Loss (src/losses/contrastive_flow_loss.py)
```python
# OLD: Weak pairwise similarity matching
# NEW: Explicit attribute prototypes with InfoNCE loss
```
**Why**: Creates explicit "target zones" for each attribute combination

---

## What These Fixes Do

### Before (Mode Collapse)
```
All inputs → Same output at [21,21]

"Smiling young woman"    → [21, 21]
"Old bearded man"        → [21, 21]
"Teenager with glasses"  → [21, 21]
```

### After (Correct Behavior - Expected)
```
Different attribute combinations → Different regions

"Smiling young woman"    → Region A (but each woman unique within region)
"Old bearded man"        → Region B (but each man unique within region)
"Teenager with glasses"  → Region C (but each teen unique within region)

Total: 10-20 distinct regions for different attribute combinations
```

---

## How to Retrain

### Step 1: Delete old checkpoint
```bash
rm -rf results/fclf_training/checkpoints/*
```
**IMPORTANT**: Old model learned bad behavior, must start fresh!

### Step 2: Train
```bash
bash scripts/train_fclf.sh
```

### Step 3: Evaluate every 5 epochs
```bash
python scripts/evaluate_attribute_transfer.py \
    --checkpoint results/fclf_training/checkpoints/fclf_latest.pt \
    --output_dir results/transfer_eval_epoch_N
```

---

## Expected Progress

| Epoch | Movement | Silhouette Target | Green Dot Regions | Status |
|-------|----------|-------------------|-------------------|---------|
| 1 (before) | 0.24 | 1.00 | 1 (collapsed) | ❌ Mode collapse |
| 5 | 0.3-0.5 | 0.6-0.8 | 3-5 | 🟡 Improving |
| 15 | 0.4-0.6 | 0.4-0.6 | 8-12 | ✅ Good! |
| 30+ | 0.5-0.7 | 0.35-0.55 | 10-20 | ✅ Converged |

**Success criteria** (15+ epochs):
- Silhouette by target: 0.35-0.55 (NOT 1.0!)
- Green dots: 10+ visible regions in UMAP plot
- No sklearn "only 1 cluster found" warnings

---

## Quick Diagnostics

### Still collapsing after 10 epochs?

1. **Check TensorBoard**:
```bash
tensorboard --logdir results/fclf_training/logs
```
- `Train/Contrastive` should decrease from ~2.0 → ~0.5
- `Train/Identity` should be similar magnitude to `Train/Contrastive`

2. **If Identity too strong**: Reduce further
```yaml
lambda_identity: 0.05 → 0.01  # In fclf_config.yaml
```

3. **If Contrastive not decreasing**: Reduce learning rate
```yaml
learning_rate: 0.0001 → 0.00005  # In fclf_config.yaml
```

---

## Why This Will Work

### Root Cause
The old `lambda_identity: 1.0` was preventing the model from learning attribute-specific flows:

```python
# Training loss
total_loss = contrastive_loss + 1.0 * identity_loss

# Identity loss dominates!
# Model learns: "Don't move much" → moves to nearest safe point (mode collapse)
```

### After Fix
```python
total_loss = contrastive_loss + 0.05 * identity_loss

# Contrastive loss dominates!
# Model learns: "Move to attribute-specific prototype" → diverse flows
```

### Prototype Loss Mechanism
```python
# For each batch:
1. Compute prototypes = mean(embeddings per attribute combo)
2. Pull embeddings toward correct prototype
3. Push embeddings away from wrong prototypes

# Result: 10-20 distinct attribute-specific clusters
```

---

## Files Modified

1. `configs/fclf_config.yaml` - Reduced lambda_identity 1.0 → 0.05
2. `src/training/train_fclf.py` - Always apply attribute augmentation
3. `src/losses/contrastive_flow_loss.py` - Prototype-based contrastive loss

See [MODE_COLLAPSE_FIXES.md](MODE_COLLAPSE_FIXES.md) for detailed explanation.

---

## Summary: The One-Sentence Fix

**Reduced identity loss from 1.0 to 0.05 and added prototype-based contrastive loss to create explicit attribute-specific target regions.**

Expected result: After 15 epochs, green dots spread into 10+ regions with silhouette ~0.4-0.6 (no more mode collapse!)
