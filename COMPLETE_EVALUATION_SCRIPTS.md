# Complete Evaluation Scripts - Quick Reference

All evaluation scripts are now implemented and ready to use!

## Scripts Overview

### 1. Training Visualization
**Script:** `scripts/plot_training_curves.py`
**Time:** 2 minutes
**Purpose:** Convert TensorBoard logs to publication-ready plots

### 2. Flow Depth Analysis ⭐ MOST IMPORTANT
**Script:** `scripts/evaluate_vs_flow_depth.py`  
**Time:** 5 minutes
**Purpose:** Test different K values, recommend optimal flow depth

### 3. Comprehensive Paper Metrics
**Script:** `scripts/compute_paper_metrics.py` (ENHANCED)
**Time:** 15 minutes
**Purpose:** All metrics including geometry, leakage, AUC, baselines

### 4. Evaluation Report Generator
**Script:** `scripts/create_evaluation_report.py`
**Time:** Instant
**Purpose:** Combine all results into single Markdown document

## Quick Start Commands

```bash
# After training completes, run these in order:

# 1. Training curves
python scripts/plot_training_curves.py \
    --logdir outputs/v4_projection/logs \
    --output_dir outputs/v4_projection/plots

# 2. Flow depth (answers "Is K=10 optimal?")
python scripts/evaluate_vs_flow_depth.py \
    --checkpoint outputs/v4_projection/checkpoints/fclf_best.pt \
    --celeba_root data/celeba \
    --embedding_dir data/embeddings \
    --output_dir outputs/v4_projection/flow_depth \
    --device cuda

# 3. Full metrics
python scripts/compute_paper_metrics.py \
    --checkpoint outputs/v4_projection/checkpoints/fclf_best.pt \
    --celeba_root data/celeba \
    --embedding_dir data/embeddings \
    --output_dir outputs/v4_projection/paper_metrics \
    --device cuda

# 4. Flipbooks
python scripts/visualize_flipbook.py \
    --flipbook_data outputs/v4_projection/paper_metrics/flipbook_data.json \
    --celeba_root data/celeba \
    --output_dir outputs/v4_projection/paper_metrics/flipbooks

# 5. Generate report
python scripts/create_evaluation_report.py \
    --base_dir outputs/v4_projection \
    --output outputs/v4_projection/EVALUATION_REPORT.md
```

## What Each Script Outputs

### plot_training_curves.py
- `training_curves.png` - 6-panel loss figure
- `loss_breakdown.png` - Stacked area chart
- `loss_ratios.png` - Component contributions
- Console: Training summary statistics

### evaluate_vs_flow_depth.py
- `flow_depth_analysis.json` - Metrics at each K
- `accuracy_vs_K.png` - Per-attribute accuracy curves
- `geometry_vs_K.png` - Clustering evolution
- Console: Optimal K recommendation

### compute_paper_metrics.py (enhanced)
- `paper_metrics.json` - All metrics including NEW geometry metrics
- `flipbook_data.json` - Visualization data
- `auc_curves.png` - AUC progression plots
- Console: Comprehensive comparison table

### create_evaluation_report.py
- `EVALUATION_REPORT.md` - Complete evaluation document

## Training Enhancement

**New feature in train_fclf.py:**
- Validation snapshots every 5 epochs
- Quick AUC check at K=5
- Logged to TensorBoard under "Snapshot/"
- Early warning for mode collapse/overfitting

## Key Questions Answered

✅ **Is K=10 optimal?** → evaluate_vs_flow_depth.py
✅ **Does projection work?** → Geometry metrics + flipbooks  
✅ **When to stop training?** → Validation snapshots
✅ **Better than baseline?** → Method comparison in metrics
✅ **Attributes interfering?** → Attribute leakage

## Documentation

- **Full details:** EVALUATION_FRAMEWORK_SUMMARY.md
- **Method changes:** METHODS.md (v4 section)
- **Quick reference:** This file

Ready to evaluate v4 model! 🚀
