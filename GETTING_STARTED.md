# Getting Started with FCLF

Quick guide to get your FCLF project up and running.

## Step-by-Step Setup

### 1. Environment Setup (5 minutes)

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### 2. Download CelebA Dataset (30 minutes)

**Option A: Manual Download**
1. Visit: https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8
2. Download `img_align_celeba.zip` (1.3GB)
3. Download `list_attr_celeba.txt`
4. Extract to `data/celeba/`

**Option B: Using gdown (if available)**
```bash
pip install gdown
mkdir -p data/celeba
cd data/celeba

# Download images (get link from Google Drive)
# Download attributes
# Extract files
```

**Verify structure:**
```
data/celeba/
├── img_align_celeba/
│   ├── 000001.jpg
│   ├── 000002.jpg
│   └── ... (202,599 images)
└── list_attr_celeba.txt
```

### 3. Precompute CLIP Embeddings (20-30 minutes)

```bash
python scripts/precompute_embeddings.py \
    --celeba_root data/celeba \
    --output_dir data/embeddings \
    --batch_size 128
```

**Expected output:**
- `data/embeddings/train_embeddings.pt` (162,000 samples)
- `data/embeddings/val_embeddings.pt` (20,000 samples)
- `data/embeddings/test_embeddings.pt` (20,599 samples)

### 4. Train Decoder (2-3 hours)

```bash
./scripts/train_decoder.sh \
    --celeba_root data/celeba \
    --embedding_dir data/embeddings \
    --output_dir outputs/decoder
```

**Monitor with TensorBoard:**
```bash
tensorboard --logdir outputs/decoder/logs
```

**What to watch:**
- Training loss should decrease steadily
- Validation loss should reach < 0.05
- Reconstructed images should be recognizable by epoch 10

**Checkpoints saved to:**
- `outputs/decoder/checkpoints/decoder_best.pt`
- `outputs/decoder/checkpoints/decoder_latest.pt`

### 5. Train FCLF Vector Field (4-6 hours)

```bash
./scripts/train_fclf.sh \
    --celeba_root data/celeba \
    --embedding_dir data/embeddings \
    --output_dir outputs/fclf
```

**Monitor with TensorBoard:**
```bash
tensorboard --logdir outputs/fclf/logs
```

**What to watch:**
- Contrastive loss should decrease
- Curl and divergence regularization should stay small
- Total loss should converge by epoch 30-40

**Checkpoints saved to:**
- `outputs/fclf/checkpoints/fclf_best.pt`
- `outputs/fclf/checkpoints/fclf_epoch_5.pt`, etc.

### 6. Evaluate (5-10 minutes)

```bash
python scripts/run_evaluation.py \
    --checkpoint outputs/fclf/checkpoints/fclf_best.pt \
    --celeba_root data/celeba \
    --embedding_dir data/embeddings \
    --output_dir results/evaluation \
    --num_samples 1000
```

**Expected metrics:**
- Silhouette score: > 0.2 (good), > 0.3 (excellent)
- Cluster purity: > 0.6
- Trajectory smoothness: Low variance

**Outputs:**
- `results/evaluation/metrics.json`
- `results/evaluation/figures/embeddings_original_umap.png`
- `results/evaluation/figures/embeddings_flowed_umap.png`
- `results/evaluation/figures/trajectories_umap.png`

### 7. Run Inference (seconds)

```bash
# Example: Make someone smile
python -m src.inference.inference \
    --image path/to/face.jpg \
    --vector_field outputs/fclf/checkpoints/fclf_best.pt \
    --decoder outputs/decoder/checkpoints/decoder_best.pt \
    --attributes "Smiling:1" \
    --output smiling.png
```

**Available attributes:**
- `Smiling`: 0 (not smiling) or 1 (smiling)
- `Young`: 0 (older) or 1 (younger)
- `Male`: 0 (female) or 1 (male)
- `Eyeglasses`: 0 (no glasses) or 1 (with glasses)
- `Mustache`: 0 (no mustache) or 1 (with mustache)

**Multi-attribute example:**
```bash
python -m src.inference.inference \
    --image input.jpg \
    --vector_field outputs/fclf/checkpoints/fclf_best.pt \
    --decoder outputs/decoder/checkpoints/decoder_best.pt \
    --attributes "Smiling:1,Young:1,Eyeglasses:0" \
    --output output.png
```

## Quick Tests

### Test 1: Verify Data Loading

```python
from src.data.celeba_dataset import CelebADataset

dataset = CelebADataset(
    root_dir='data/celeba',
    split='train'
)

print(f"Dataset size: {len(dataset)}")
sample = dataset[0]
print(f"Attributes: {sample['attributes']}")
print(f"Image path: {sample['image_path']}")
```

### Test 2: Verify Embeddings

```python
import torch

embeddings = torch.load('data/embeddings/train_embeddings.pt')
print(f"Embeddings shape: {embeddings.shape}")  # Should be [162000, 512]
print(f"Embedding stats: mean={embeddings.mean():.4f}, std={embeddings.std():.4f}")
```

### Test 3: Verify Models

```python
from src.models.vector_field import VectorFieldNetwork

model = VectorFieldNetwork(embedding_dim=512, num_attributes=5, hidden_dim=256)
z = torch.randn(8, 512)
y = torch.randint(0, 2, (8, 5)).float()

v = model(z, y)
print(f"Velocity shape: {v.shape}")  # Should be [8, 512]
```

## Troubleshooting

### Issue: CUDA out of memory

**Solution:**
```yaml
# In configs/decoder_config.yaml
training:
  batch_size: 32  # Reduce from 64

# In configs/fclf_config.yaml
training:
  batch_size: 64  # Reduce from 128
```

### Issue: CLIP not installing

**Solution:**
```bash
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```

### Issue: Decoder quality poor

**Solutions:**
1. Train for more epochs (increase from 20 to 30)
2. Check validation loss is decreasing
3. Try VAE decoder: `./scripts/train_decoder.sh --use_vae`

### Issue: FCLF loss not decreasing

**Solutions:**
1. Disable regularization first: `./scripts/train_fclf.sh --no_regularization`
2. Reduce learning rate in config (1e-4 → 5e-5)
3. Verify embeddings are normalized

## Next Steps

### Experiments to Try

1. **Ablation Studies**
   - Train without regularization
   - Vary temperature parameter
   - Different flow step sizes

2. **Hyperparameter Tuning**
   - Learning rates: [1e-5, 5e-5, 1e-4, 5e-4]
   - Alpha (flow step): [0.05, 0.1, 0.2, 0.5]
   - Hidden dimensions: [128, 256, 512]

3. **Advanced Features**
   - Multi-attribute flows
   - Continuous attribute control
   - Interpolation between attributes

### Notebooks to Create

1. `01_data_exploration.ipynb`: Analyze attribute distributions
2. `02_decoder_analysis.ipynb`: Visualize reconstructions
3. `03_fclf_results.ipynb`: Visualize flows and trajectories
4. `04_ablation_studies.ipynb`: Compare different configurations

## Timeline

**Week 1**: Setup + Data + Decoder Training
**Week 2**: FCLF Training + Initial Evaluation
**Week 3**: Hyperparameter tuning + Ablations
**Week 4**: Final experiments + Visualization
**Week 5**: Report writing + Presentation prep

## Resources

- **TensorBoard**: http://localhost:6006
- **CLIP Paper**: https://arxiv.org/abs/2103.00020
- **CelebA Paper**: https://arxiv.org/abs/1411.7766

## Support

If you encounter issues:
1. Check this guide first
2. Review the main [README.md](README.md)
3. Consult the [PRD.md](PRD.md) for detailed specifications
4. Check TensorBoard logs for training issues

Good luck with your FCLF implementation! 🚀
