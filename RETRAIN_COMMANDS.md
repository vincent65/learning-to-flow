# Quick Commands for Retraining and Evaluation

## Step 1: Pull the Fix to Your VM

```bash
cd ~/learning-to-flow
git reset --hard
git fetch origin
git reset --hard origin/main
```

## Step 2: Verify Fixes Are Applied

```bash
# Check that the fixes are in place
grep -n "similarity_threshold" src/losses/contrastive_flow_loss.py
grep -n "lambda_contrastive: 0.2" configs/fclf_config.yaml
grep -n "lambda_identity: 0.8" configs/fclf_config.yaml
```

You should see output confirming all three fixes are present.

## Step 3: Retrain the Model (~2 hours)

```bash
python src/training/train_fclf.py \
    --config configs/fclf_config.yaml \
    --celeba_root data/celeba \
    --embedding_dir data/embeddings \
    --output_dir results/fixed_model \
    --device cuda
```

**Training time:** ~2 hours for 50 epochs on L4 GPU

**To train for fewer epochs (faster testing):**
```bash
# Just 10 epochs for quick testing (~25 minutes)
python src/training/train_fclf.py \
    --config configs/fclf_config.yaml \
    --celeba_root data/celeba \
    --embedding_dir data/embeddings \
    --output_dir results/fixed_model \
    --device cuda \
    --epochs 10
```

## Step 4: Re-Evaluate with Fixed Model (~10 minutes)

```bash
python scripts/compute_paper_metrics.py \
    --checkpoint results/fixed_model/checkpoints/fclf_best.pt \
    --celeba_root data/celeba \
    --embedding_dir data/embeddings \
    --output_dir paper_metrics_fixed \
    --device cuda
```

**This automatically generates:**
- `paper_metrics_fixed/paper_metrics.json` - All quantitative metrics
- `paper_metrics_fixed/flipbook_data.json` - Data for flipbook visualizations
- `paper_metrics_fixed/auc_curves.png` - AUC along flow path for each attribute (5 plots)

## Step 5: Generate LaTeX Tables

```bash
python scripts/generate_latex_tables.py paper_metrics_fixed/paper_metrics.json
```

Output will be saved to `paper_metrics_fixed/paper_metrics_tables.tex`

## Step 6: Generate Flipbook Visualizations

```bash
python scripts/visualize_flipbook.py \
    --flipbook_data paper_metrics_fixed/flipbook_data.json \
    --celeba_root data/celeba \
    --output_dir paper_metrics_fixed/flipbooks \
    --num_flipbooks 20
```

## Step 7: Generate UMAP Trajectory Visualizations (Optional)

```bash
python scripts/evaluate_attribute_transfer.py \
    --checkpoint results/fixed_model/checkpoints/fclf_best.pt \
    --celeba_root data/celeba \
    --embedding_dir data/embeddings \
    --output_dir evaluation_results \
    --num_samples 1000 \
    --device cuda
```

This generates:
- UMAP plots colored by original attributes
- UMAP plots colored by target attributes
- 2D trajectory visualizations
- Saved to `evaluation_results/figures/`

---

## Quick Test (If You're in a Hurry)

If you just want to quickly verify the fix works:

```bash
# Train for only 5 epochs (~12 minutes)
python src/training/train_fclf.py \
    --config configs/fclf_config.yaml \
    --celeba_root data/celeba \
    --embedding_dir data/embeddings \
    --output_dir results/quick_test \
    --device cuda \
    --epochs 5

# Evaluate
python scripts/compute_paper_metrics.py \
    --checkpoint results/quick_test/checkpoints/fclf_epoch_5.pt \
    --celeba_root data/celeba \
    --embedding_dir data/embeddings \
    --output_dir test_metrics \
    --device cuda \
    --num_samples 500

# Generate flipbooks (just 5 to check)
python scripts/visualize_flipbook.py \
    --flipbook_data test_metrics/flipbook_data.json \
    --celeba_root data/celeba \
    --output_dir test_metrics/flipbooks \
    --num_flipbooks 5
```

Then look at the flipbooks in `test_metrics/flipbooks/` - they should show diverse faces with gradual attribute changes, NOT all converging to "person in suit".

---

## All Generated Files Summary

After running the complete pipeline, you'll have:

### Quantitative Metrics:
- `paper_metrics_fixed/paper_metrics.json` - All metrics in JSON format
- `paper_metrics_fixed/paper_metrics_tables.tex` - LaTeX tables for paper

### Visualizations:
- `paper_metrics_fixed/auc_curves.png` - **AUC trajectory plots** (5 subplots showing monotonic progress)
- `paper_metrics_fixed/flipbooks/flipbook_XXXX_*.png` - Individual flipbooks with descriptive names
- `paper_metrics_fixed/flipbooks/flipbook_summary.png` - Summary montage of 5 flipbooks
- `evaluation_results/figures/*.png` - UMAP visualizations (if you run Step 7)

## Expected Differences After Retraining

| Metric | OLD (collapsed) | NEW (fixed) | Status |
|--------|----------------|-------------|--------|
| Flipbook diversity | ❌ All → same face | ✅ Diverse faces | **Main fix** |
| k-NN Purity | 96.2% | ~85-90% | Slightly lower, OK |
| Centroid Distance | 0.039 | ~0.06-0.09 | Slightly higher, OK |
| AUC | 0.80-0.86 | ~0.75-0.82 | Maintained |
| Visual quality | ❌ Wrong semantics | ✅ Correct attributes | **Main fix** |

---

## Troubleshooting

### If you get "ModuleNotFoundError":
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### If you get "FileNotFoundError" for embeddings:
```bash
ls data/embeddings/
# Should see: train_embeddings.pt, val_embeddings.pt, test_embeddings.pt
```

### If you get "CUDA out of memory":
Edit `configs/fclf_config.yaml` and reduce batch_size:
```yaml
training:
  batch_size: 256  # Reduce from 512
```

### To monitor training progress:
```bash
# In another terminal
watch -n 5 'ls -lh results/fixed_model/checkpoints/'
```

---

## Full Pipeline (Copy-Paste Ready)

```bash
#!/bin/bash
# Complete pipeline to retrain and evaluate

cd ~/learning-to-flow

# Pull fixes
git reset --hard origin/main

# Retrain (20 epochs)
python src/training/train_fclf.py \
    --config configs/fclf_config.yaml \
    --celeba_root data/celeba \
    --embedding_dir data/embeddings \
    --output_dir results/fixed_model \
    --device cuda \
    --epochs 20

# Evaluate
python scripts/compute_paper_metrics.py \
    --checkpoint results/fixed_model/checkpoints/fclf_best.pt \
    --celeba_root data/celeba \
    --embedding_dir data/embeddings \
    --output_dir paper_metrics_fixed \
    --device cuda

# Generate tables
python scripts/generate_latex_tables.py paper_metrics_fixed/paper_metrics.json

# Generate flipbooks
python scripts/visualize_flipbook.py \
    --flipbook_data paper_metrics_fixed/flipbook_data.json \
    --celeba_root data/celeba \
    --output_dir paper_metrics_fixed/flipbooks \
    --num_flipbooks 20

echo "✅ Complete! Check paper_metrics_fixed/ for all results"
```

Save this as `retrain.sh`, then:
```bash
chmod +x retrain.sh
./retrain.sh
```
