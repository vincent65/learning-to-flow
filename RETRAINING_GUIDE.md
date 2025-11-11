# Retraining Guide: Fixing Mode Collapse

## What Was Wrong

Your initial training had **severe mode collapse** - the model collapsed all 5,000 embeddings to just 10-15 discrete points. This happened because:

1. **No identity preservation**: Nothing prevented the model from mapping everything to the same point
2. **Wrong training objective**: Model was trained to flow embeddings toward their EXISTING attributes, not to CHANGE attributes
3. **Weak regularization**: Curl/divergence penalties weren't strong enough

### Symptoms of Mode Collapse

- Silhouette score = 1.0 (suspiciously perfect)
- Flowed embeddings cluster in ~10-15 discrete points
- sklearn warning: "Number of distinct clusters (1) found"
- Huge trajectory steps (mean=0.71, should be ~0.03-0.05)

---

## What Changed

### 1. Identity Preservation Loss

**Added to [src/losses/combined_loss.py](src/losses/combined_loss.py):**

```python
# New loss term prevents mode collapse
identity_loss = torch.mean((z_flowed - z) ** 2)

total_loss = (
    contrastive_loss +
    lambda_curl * curl_loss +
    lambda_div * div_loss +
    lambda_identity * identity_loss  # NEW!
)
```

**What it does:** Penalizes large deviations from the original embedding. This prevents the model from pushing everything to the same point.

**Weight:** `lambda_identity = 0.1` (in config)

### 2. Attribute Augmentation During Training

**Added to [src/training/train_fclf.py](src/training/train_fclf.py):**

```python
# NEW: Randomly flip 1-2 attributes 50% of the time
target_attributes = attributes.clone()
if torch.rand(1).item() > 0.5:
    for i in range(batch_size):
        num_flips = torch.randint(1, 3, (1,)).item()  # Flip 1 or 2
        attrs_to_flip = torch.randperm(5)[:num_flips]
        for attr_idx in attrs_to_flip:
            target_attributes[i, attr_idx] = 1 - target_attributes[i, attr_idx]

# Flow toward TARGET attributes, not original
loss = criterion(model, embeddings, target_attributes)
```

**What it does:**
- **50% of batches**: Keep original attributes (learn to preserve)
- **50% of batches**: Flip 1-2 attributes (learn to transfer)

This teaches the model to actually CHANGE attributes rather than just "clean up" embeddings.

### 3. Updated Config

**[configs/fclf_config.yaml](configs/fclf_config.yaml):**

```yaml
loss:
  temperature: 0.07
  lambda_curl: 0.01
  lambda_div: 0.01
  lambda_identity: 0.1  # NEW parameter
```

---

## How to Retrain

### Step 1: Clear Old Checkpoints (Optional but Recommended)

```bash
# On your VM
rm -rf outputs/fclf/checkpoints/*
rm -rf outputs/fclf/logs/*
```

This ensures you start fresh and don't accidentally resume from the collapsed model.

### Step 2: Start Training

```bash
# Quick test (10 epochs, ~4 hours on T4)
./scripts/train_fclf.sh \
    --celeba_root data/celeba \
    --embedding_dir data/embeddings \
    --output_dir outputs/fclf
```

Or edit the config to reduce epochs for faster testing:

```yaml
# In configs/fclf_config.yaml
training:
  num_epochs: 10  # Change from 50 for quick test
```

### Step 3: Monitor Training

```bash
# In a separate terminal
tensorboard --logdir outputs/fclf/logs
```

**What to watch:**

```
Progress bar should show:
loss: 2.345  contr: 2.100  curl: 0.012  div: 0.008  ident: 0.015
                                                      ↑ NEW!
```

**Good signs:**
- Contrastive loss decreasing
- Identity loss staying small (~0.01-0.02)
- Curl/div losses staying small
- No sudden jumps

**Bad signs:**
- Identity loss growing large (>0.1) = too much movement
- Contrastive loss not decreasing = not learning
- Loss = NaN = training collapsed

---

## Evaluating the New Model

### Use the Attribute Transfer Evaluation (Correct Method)

```bash
python scripts/evaluate_attribute_transfer.py \
    --checkpoint outputs/fclf/checkpoints/fclf_best.pt \
    --celeba_root data/celeba \
    --embedding_dir data/embeddings \
    --output_dir results/transfer_eval_fixed \
    --num_samples 5000 \
    --transfer_mode single
```

### What to Look For

**SUCCESS (Model is Working):**

```
Silhouette Scores:
  Flowed embeddings clustered by original attrs:   0.10-0.25 (LOW - good!)
  Flowed embeddings clustered by target attrs:     0.35-0.55 (HIGH - good!)

✓ Flowed embeddings match TARGET attributes!

Trajectory Smoothness:
  Mean step distance: 0.025-0.045 (small, consistent steps)
  Std step distance: 0.008-0.015 (low variance)
```

**Visualizations:**
- `flowed_by_target_attrs.png` should show CONTINUOUS, well-separated clusters
- `transfer_trajectories.png` should show smooth, consistent arrows
- No more discrete point collapse!

**FAILURE (Still Collapsed):**

```
Silhouette Scores:
  Both near 1.0

Cluster Purity:
  Both very high (>0.85)

Warning: "Number of distinct clusters (1) found"
```

If this happens:
- Increase `lambda_identity` from 0.1 to 0.2
- Decrease batch size (helps with stability)
- Check TensorBoard - is training actually progressing?

---

## Expected Timeline

**10 epochs on T4 GPU:**
- Training time: ~4 hours
- You should see clear improvement in metrics
- Good enough for validation

**50 epochs on T4 GPU:**
- Training time: ~20 hours
- Near-optimal performance
- Use for final results/paper

---

## Hyperparameter Tuning (If Needed)

If after 10 epochs you still see issues:

### If Mode Collapse Persists:

```yaml
loss:
  lambda_identity: 0.2  # Increase from 0.1 (stronger identity preservation)
  lambda_curl: 0.05     # Increase from 0.01 (smoother flows)
  lambda_div: 0.05      # Increase from 0.01
```

### If Model Not Learning (Loss Not Decreasing):

```yaml
training:
  learning_rate: 0.0005  # Increase from 0.0001

loss:
  lambda_identity: 0.05  # Decrease from 0.1 (less constraint)
```

### If Training Unstable (NaN Loss):

```yaml
training:
  learning_rate: 0.00005  # Decrease from 0.0001
  batch_size: 64          # Decrease from 128

loss:
  temperature: 0.1  # Increase from 0.07 (softer contrastive loss)
```

---

## Comparison: Before vs After

### Before (Mode Collapse):

```
Training:
  - Flowed embeddings toward EXISTING attributes
  - No identity preservation

Evaluation:
  - 10-15 discrete output points
  - Silhouette = 1.0 (collapsed)
  - Mean step = 0.71 (huge jumps)

Behavior:
  - Ignores target attributes
  - Just maps to lookup table
```

### After (Fixed):

```
Training:
  - Flowed embeddings toward FLIPPED attributes (50% of time)
  - Identity loss prevents collapse

Evaluation (Expected):
  - Continuous embedding space
  - Silhouette = 0.35-0.55 (good clustering)
  - Mean step = 0.03-0.05 (smooth flows)

Behavior:
  - Actually transfers attributes
  - Smooth, continuous transformations
```

---

## Troubleshooting

### Q: Training is very slow

**A:** Attribute augmentation adds minimal overhead. If it's much slower:
- Check if you accidentally left `num_workers > 0` (should be 0 on VM)
- Reduce batch size if GPU memory is maxed out

### Q: Loss is oscillating wildly

**A:** Reduce learning rate:
```yaml
training:
  learning_rate: 0.00005
```

### Q: Identity loss keeps growing

**A:** Model is trying to move embeddings too far. Increase lambda_identity:
```yaml
loss:
  lambda_identity: 0.2
```

### Q: After retraining, still seeing discrete points

**A:** Mode collapse is stubborn. Try:
1. Verify you're using the NEW checkpoints, not old ones
2. Increase `lambda_identity` to 0.3
3. Train for more epochs (convergence takes time)
4. Check if contrastive loss is actually decreasing

---

## Next Steps After Successful Retraining

1. **Run full evaluation** on test set (20,599 samples)
2. **Try inference** on real images with attribute manipulation
3. **Generate visualizations** for your report
4. **Compare quantitative metrics** before/after in your writeup

Good luck! 🚀
