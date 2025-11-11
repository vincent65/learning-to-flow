# Critical Fixes Applied to FCLF Implementation

## Summary

Your FCLF implementation had **three fundamental issues** that have now been fixed:

1. ✅ **Missing Normalization** - Flowed embeddings were leaving CLIP space
2. ✅ **Zero Contrastive Loss** - Loss function returned zero too frequently
3. ✅ **Weak Identity Constraint** - Insufficient penalty for excessive movement

All fixes have been committed and pushed to GitHub.

---

## Fix #1: Normalization Constraint (CRITICAL)

### The Problem

```python
# BEFORE (BROKEN):
v = vector_field(z, y)
z_flowed = z + 0.1 * v  # Can have ANY magnitude!
```

**What was wrong:**
- CLIP embeddings are **unit-normalized** (norm = 1.0)
- Your vector field could output arbitrarily large velocities
- After flowing: `z_flowed` had norm ≈ 8-10 (completely invalid!)
- These were **NOT valid CLIP embeddings** anymore

**Evidence:**
```
Mean distance moved: 8.1186  ← Should be <1.5!
```

### The Fix

```python
# AFTER (FIXED):
v = vector_field(z, y)
z_flowed = z + 0.1 * v
z_flowed = F.normalize(z_flowed, dim=1)  # Force norm=1.0
```

**File:** `src/losses/combined_loss.py` line 92

**Why this matters:**
- Keeps embeddings on the unit hypersphere (valid CLIP space)
- Identity loss now measures **angular distance** (correct for unit sphere)
- Movement is bounded to 0-2.0 range (can't exceed sphere diameter)
- All downstream tools (decoder, evaluation) expect normalized embeddings

---

## Fix #2: Contrastive Loss Function

### The Problem

```python
# SimpleContrastiveLoss (OLD):
# Returns 0.0 if no samples have EXACTLY matching attributes
# With attribute augmentation → many batches have no exact matches → loss=0

if count == 0:
    return 0.0  # No learning signal!
```

**What was wrong:**
- With 5 binary attributes, you need ALL 5 to match for a "positive pair"
- Attribute augmentation randomly flips 1-2 attributes
- Result: Many batches have **all unique attribute combinations**
- When loss=0: **No gradient for attribute manipulation!**

**Evidence:**
During training you likely saw:
```
loss: 2.3  contr: 1.5  ...
loss: 1.8  contr: 0.0  ... ← ZERO!
loss: 2.1  contr: 1.2  ...
loss: 1.9  contr: 0.0  ... ← ZERO!
```

### The Fix

```python
# AttributeWeightedContrastiveLoss (NEW):
# Uses soft similarity based on number of matching attributes
# NEVER returns zero - always provides gradient signal

attr_similarity = torch.matmul(attributes, attributes.t())
attr_weights = attr_similarity / num_attrs  # 0 to 1 scale
# 0/5 match → weight=0.0, 3/5 match → weight=0.6, 5/5 match → weight=1.0
```

**File:** `src/losses/combined_loss.py` line 52

**Why this matters:**
- **Consistent gradient signal** in every batch
- Rewards partial attribute matches (more natural for continuous changes)
- Model learns smooth interpolation between attribute combinations
- No more degenerate zero-loss batches

---

## Fix #3: Regularization Weights

### The Problem

```yaml
# OLD CONFIG:
loss:
  lambda_curl: 0.01
  lambda_div: 0.01
  lambda_identity: 0.1  # TOO WEAK!
```

**What was wrong:**
- Contrastive loss ≈ 2.0 (dominates)
- Identity loss = 0.1 * (8.1)² ≈ 6.5 (still too weak)
- Model learned: "Achieve perfect clustering at ANY cost, ignore distance"

### The Fix

```yaml
# NEW CONFIG:
loss:
  lambda_curl: 0.05      # 5x stronger
  lambda_div: 0.05       # 5x stronger
  lambda_identity: 1.0   # 10x stronger
```

**File:** `configs/fclf_config.yaml`

**Why this matters:**
- Identity loss now has equal weight with contrastive loss
- Model must balance clustering quality vs movement distance
- Combined with normalization, prevents excessive movement
- Results in smooth, bounded transformations

---

## How These Fixes Work Together

### Before (Broken):

```python
# Model learns:
1. Contrastive loss says: "Cluster perfectly!" (sometimes zero signal)
2. Identity loss weakly says: "Don't move too much" (weight=0.1)
3. No normalization constraint

# Result:
- Model throws embeddings 8+ units away
- Leaves CLIP embedding space entirely
- Training unstable (zero-loss batches)
- Mode collapse or erratic behavior
```

### After (Fixed):

```python
# Model learns:
1. Contrastive loss says: "Cluster by attributes" (always provides signal)
2. Identity loss strongly says: "Stay close to original" (weight=1.0)
3. Normalization forces valid CLIP embeddings

# Result:
- Embeddings stay on unit sphere (valid CLIP space)
- Smooth, small movements (0.2-0.6 units)
- Consistent gradient signal every batch
- Stable, continuous transformations
```

---

## What to Expect After Retraining

### Training Behavior

**Progress bar should show:**
```
Epoch 1/50:
loss: 0.45  contr: 0.20  curl: 0.02  div: 0.02  ident: 0.03
loss: 0.43  contr: 0.18  curl: 0.02  div: 0.01  ident: 0.04
loss: 0.41  contr: 0.16  curl: 0.02  div: 0.02  ident: 0.03
```

**Key indicators:**
- ✅ Contrastive loss **never zero** (always 0.1-0.3 range)
- ✅ Identity loss small (0.01-0.05 = small angular movement)
- ✅ Total loss steadily decreasing
- ✅ No NaN or inf values

### Evaluation Metrics

**Expected Results:**

```
Silhouette Scores:
  Flowed by original attrs:  0.10-0.25  (low - attributes changed)
  Flowed by target attrs:    0.30-0.50  (moderate clustering)

✓ NOT 1.0! Continuous space preserved.

Cluster Purity:
  Original: 0.27
  Flowed:   0.50-0.70 (clear improvement)

Trajectory Smoothness:
  Mean step: 0.03-0.08 (smooth, consistent)
  Std step:  0.01-0.03 (low variance)

Embedding Movement:
  Mean distance: 0.2-0.6 (reasonable angular distance)
  Max distance: <1.4 (physically bounded by sphere)

✓ NO sklearn warnings about "only 1 cluster"!
```

### Visualizations

**Flowed Embeddings by Target Attributes:**
- Continuous, smooth point distribution
- Visible clustering by attribute (but not perfect)
- No discrete point collapse

**Trajectories:**
- Smooth curves (not zigzagging)
- Evenly spaced intermediate steps
- Multiple diverse endpoints

---

## Retraining Instructions

### Step 1: On Your VM, Pull Latest Code

```bash
cd ~/learning-to-flow
git pull
```

You should see:
```
Updating 865c125..3a11118
Fast-forward
 src/losses/combined_loss.py | 13 +++++++------
 configs/fclf_config.yaml     |  6 +++---
 2 files changed, 10 insertions(+), 9 deletions(-)
```

### Step 2: Clear Old Checkpoints

Your old model was trained with broken constraints. Start fresh:

```bash
rm -rf outputs/fclf/checkpoints/*
rm -rf outputs/fclf/logs/*
```

### Step 3: Start Training

```bash
# Quick test (10 epochs ≈ 4 hours)
./scripts/train_fclf.sh \
    --celeba_root data/celeba \
    --embedding_dir data/embeddings \
    --output_dir outputs/fclf
```

Or for full training, edit the config:
```yaml
# configs/fclf_config.yaml
training:
  num_epochs: 50  # Full training (~20 hours on T4)
```

### Step 4: Monitor Training

**Watch TensorBoard:**
```bash
tensorboard --logdir outputs/fclf/logs
```

**Key things to watch:**
1. **Contrastive loss**: Should be 0.15-0.30 (never zero!)
2. **Identity loss**: Should be 0.01-0.05 (small movements)
3. **Total loss**: Should decrease smoothly
4. **No sudden jumps or NaN**

### Step 5: Evaluate

After 10 epochs, run the attribute transfer evaluation:

```bash
python scripts/evaluate_attribute_transfer.py \
    --checkpoint outputs/fclf/checkpoints/fclf_best.pt \
    --celeba_root data/celeba \
    --embedding_dir data/embeddings \
    --output_dir results/transfer_eval_v2 \
    --num_samples 5000 \
    --transfer_mode single
```

**Success criteria:**
- ✅ Movement distance: 0.2-0.6 (down from 8.1!)
- ✅ Silhouette: 0.3-0.5 (not 1.0!)
- ✅ No sklearn warnings
- ✅ Smooth trajectories

---

## Troubleshooting

### If Training is Slow

Normal on T4:
- ~23 minutes per epoch
- 10 epochs ≈ 4 hours
- 50 epochs ≈ 20 hours

### If Loss is Not Decreasing

Check TensorBoard:
- Is contrastive loss always zero? (shouldn't be with new fix)
- Is identity loss very large (>0.5)? Might need to reduce lambda_identity
- Are there NaN values? Reduce learning rate

### If Still Seeing Mode Collapse After 10 Epochs

Try increasing identity loss weight:
```yaml
loss:
  lambda_identity: 2.0  # Even stronger constraint
```

### If Movement is Too Small (< 0.1)

Model being too conservative. Reduce identity loss:
```yaml
loss:
  lambda_identity: 0.5  # Allow more movement
```

---

## Summary of Changes

| Component | Before | After | Impact |
|-----------|--------|-------|--------|
| **Normalization** | None | `F.normalize(z_flowed)` | Keeps embeddings valid |
| **Contrastive Loss** | SimpleContrastiveLoss | AttributeWeightedContrastiveLoss | No more zero loss |
| **lambda_identity** | 0.1 | 1.0 | Stronger movement constraint |
| **lambda_curl** | 0.01 | 0.05 | Smoother flows |
| **lambda_div** | 0.01 | 0.05 | Smoother flows |

**All fixes are backward-compatible** - no changes to model architecture, just loss computation.

---

## Expected Timeline

- **Pull code + setup**: 5 minutes
- **Training (10 epochs)**: ~4 hours
- **Evaluation**: ~5 minutes
- **Total**: ~4-5 hours to validate fixes

If 10-epoch results look good, continue to 50 epochs for final model.

---

## Questions?

If after retraining you still see issues:
1. Share the training progress bar output
2. Share evaluation metrics
3. Share a tensorboard screenshot

These fixes address the **fundamental implementation errors**. With proper constraints, FCLF should work correctly.

Good luck! 🚀
