# Attribute-Specific Vector Field - Quick Start Guide

This guide helps you train and evaluate the **attribute-specific vector field** model, which solves the mode collapse problem by learning separate vector fields for each attribute.

## Key Differences from Original FCLF

| Aspect | Original FCLF | Attribute-Specific |
|--------|---------------|-------------------|
| **Architecture** | Single vector field for all attributes | Separate vector field per attribute |
| **Loss** | InfoNCE on 2^N combinations | Per-attribute contrastive loss |
| **Clustering** | Discrete 32-point clusters | Continuous manifolds |
| **Mode Collapse** | Severe (within-dist ~0.02) | Prevented by design |
| **Attribute Independence** | Not enforced | Orthogonality loss |

---

## File Organization

All attribute-specific files use the `attr_specific_` prefix:

```
src/
├── models/
│   ├── vector_field.py              # Original model
│   └── attr_specific_vector_field.py  # NEW: Attribute-specific model
├── losses/
│   ├── contrastive_flow_loss.py     # Original loss
│   └── attr_specific_losses.py       # NEW: Per-attribute losses
├── training/
│   ├── train_fclf.py                # Original training
│   └── train_attr_specific.py        # NEW: Attribute-specific training

configs/
├── fclf_config.yaml                 # Original config
└── attr_specific_config.yaml         # NEW: Attribute-specific config

scripts/
├── compute_paper_metrics.py         # Original evaluation
└── evaluate_attr_specific.py         # NEW: Attribute-specific evaluation
```

---

## Step-by-Step Training

### Step 1: Test on Small Subset (5 minutes)

First, verify everything works with a small test:

```bash
# Create test config
cat > configs/attr_specific_test.yaml << EOF
model:
  embedding_dim: 512
  num_attributes: 5
  hidden_dim: 256
  projection_radius: 1.0

data:
  celeba_root: "data/celeba"
  embedding_dir: "data/embeddings"

training:
  num_epochs: 2  # Just 2 epochs for testing
  batch_size: 128  # Smaller batch
  learning_rate: 0.0001
  alpha: 0.12
  flow_steps: 10

loss:
  temperature: 0.2
  lambda_contrastive: 0.5
  lambda_orthogonal: 0.1
  lambda_identity: 0.2
  lambda_smoothness: 0.1
  lambda_curl: 0.0
  lambda_div: 0.0

inference:
  num_flow_steps: 10
  step_size: 0.1
EOF

# Run test training
python src/training/train_attr_specific.py \
    --config configs/attr_specific_test.yaml \
    --output_dir outputs/attr_specific_test \
    --device cuda
```

**Expected output:**
- Should complete 2 epochs in ~5 minutes
- Loss should decrease from ~1.0 to ~0.5
- No errors or crashes

**If this works, proceed to full training!**

---

### Step 2: Full Training (10-12 hours)

```bash
# Train with full config
python src/training/train_attr_specific.py \
    --config configs/attr_specific_config.yaml \
    --output_dir outputs/attr_specific \
    --device cuda
```

**Monitor training:**

```bash
# In another terminal
tail -f outputs/attr_specific/logs/events.out.tfevents.*

# Or use TensorBoard
tensorboard --logdir outputs/attr_specific/logs
```

**Training metrics to watch:**

```
Good signs:
  - Total loss decreases smoothly (1.0 → 0.3)
  - Contrastive loss decreases (0.8 → 0.2)
  - Orthogonal loss stays low (<0.05)
  - Identity loss stays moderate (0.1-0.3)
  - No divergence explosion

Bad signs:
  - Loss plateaus early (stuck in local minimum)
  - Orthogonal loss grows (attributes interfering)
  - Divergence starts growing exponentially
```

---

### Step 3: Evaluate Model

```bash
python scripts/evaluate_attr_specific.py \
    --checkpoint outputs/attr_specific/checkpoints/best.pt \
    --embedding_dir data/embeddings \
    --celeba_root data/celeba \
    --output_dir outputs/attr_specific/evaluation \
    --num_samples 2000 \
    --device cuda
```

**Expected improvements over original:**

```
Metric                    Original    Attr-Specific    Target
─────────────────────────────────────────────────────────────
Within-class distance     0.02        0.3-0.5          0.3-0.5  ✓
Geometry ratio            26-470      5-15             5-15     ✓
Monotonic AUC             0-20%       60-90%           >50%     ✓
Flipbook collapse         Severe      Minimal          None     ✓
Linear probe accuracy     0.74        0.85-0.90        >0.85    ✓
```

---

### Step 4: Generate Flipbooks

```bash
python scripts/visualize_flipbook.py \
    --flipbook_data outputs/attr_specific/evaluation/flipbook_data.json \
    --celeba_root data/celeba \
    --output_dir outputs/attr_specific/evaluation/flipbooks \
    --num_flipbooks 20
```

**Check for success:**

```bash
# Check nearest neighbor diversity
python << EOF
import json

with open('outputs/attr_specific/evaluation/flipbook_data.json') as f:
    data = json.load(f)

for i in range(min(5, len(data))):
    fb = data[i]
    indices = [idx[0] for idx in fb['nearest_indices']]
    unique = len(set(indices))
    total = len(indices)
    status = "✓ GOOD" if unique >= total * 0.7 else "✗ COLLAPSED"
    print(f"{i}: {fb['change_string']}")
    print(f"   Unique indices: {unique}/{total} {status}")
EOF
```

**Good result:** Each flipbook should have >70% unique indices (e.g., 8/11 or better)
**Bad result:** Repeated indices like original (2/11 unique)

---

## Troubleshooting

### Issue 1: Training Loss Not Decreasing

**Symptoms:** Loss stays ~1.0 for many epochs

**Solutions:**
1. Lower learning rate: `learning_rate: 0.00005`
2. Increase contrastive weight: `lambda_contrastive: 0.8`
3. Check data is loading correctly

### Issue 2: Orthogonal Loss Growing

**Symptoms:** Orthogonal loss > 0.1 and growing

**Solutions:**
1. Increase orthogonal weight: `lambda_orthogonal: 0.2`
2. Decrease alpha (smaller steps): `alpha: 0.08`

### Issue 3: Still Mode Collapse

**Symptoms:** Flipbooks still show repeated indices

**Solutions:**
1. Increase identity loss: `lambda_identity: 0.4`
2. Increase smoothness loss: `lambda_smoothness: 0.2`
3. Use softer temperature: `temperature: 0.3`

### Issue 4: Out of Memory

**Symptoms:** CUDA out of memory error

**Solutions:**
```yaml
training:
  batch_size: 256  # Reduce from 512
  flow_steps: 5    # Reduce from 10
```

---

## Comparison with Original

### Run Both and Compare

```bash
# Evaluate original model
python scripts/compute_paper_metrics.py \
    --checkpoint checkpoints/fclf_best.pt \
    --embedding_dir data/embeddings \
    --output_dir outputs/original_comparison \
    --num_samples 2000 \
    --device cuda

# Evaluate attribute-specific model
python scripts/evaluate_attr_specific.py \
    --checkpoint outputs/attr_specific/checkpoints/best.pt \
    --embedding_dir data/embeddings \
    --output_dir outputs/attr_specific_comparison \
    --num_samples 2000 \
    --device cuda

# Compare metrics
python << EOF
import json

with open('outputs/original_comparison/paper_metrics.json') as f:
    orig = json.load(f)
with open('outputs/attr_specific_comparison/metrics.json') as f:
    attr = json.load(f)

print("Comparison:")
print(f"  Within-class:  {orig['comparison']['fclf']['within_class_dist']:.4f} → {attr['comparison']['attr_specific']['within_class_dist']:.4f}")
print(f"  Geometry ratio: {orig['comparison']['fclf']['geometry_ratio']:.2f} → {attr['comparison']['attr_specific']['geometry_ratio']:.2f}")
EOF
```

---

## Expected Timeline

- **Testing (Step 1):** 5 minutes
- **Training (Step 2):** 10-12 hours (can run overnight)
- **Evaluation (Step 3):** 30 minutes
- **Flipbooks (Step 4):** 5 minutes
- **Total:** ~11-13 hours

---

## Key Hyperparameters

If you want to adjust the model, these are the most important:

### For Better Clustering
```yaml
lambda_contrastive: 0.7  # Increase (was 0.5)
temperature: 0.15        # Decrease (was 0.2)
```

### For Better Manifold Preservation
```yaml
lambda_identity: 0.3     # Increase (was 0.2)
lambda_smoothness: 0.15  # Increase (was 0.1)
```

### For Better Attribute Independence
```yaml
lambda_orthogonal: 0.2   # Increase (was 0.1)
```

---

## Success Criteria

Your model is working well if:

1. ✓ **Within-class distance: 0.3-0.5** (not <0.1)
2. ✓ **Geometry ratio: 5-15** (not >20)
3. ✓ **Monotonic AUC: >60%** for most attributes
4. ✓ **Flipbook indices: >70% unique** per trajectory
5. ✓ **Visual inspection:** Attributes actually change in flipbooks!

The most important is #5 - open the flipbook images and verify with your eyes that attributes are changing!

---

## Next Steps

After successful training:

1. Generate more flipbooks (50-100) for best examples
2. Create comparison figures (original vs. attr-specific)
3. Write up results explaining why attr-specific works better
4. Optional: Try individual attribute flows with `flow_single_attribute_trajectory()`

Good luck! 🚀
