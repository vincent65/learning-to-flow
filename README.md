# Function-Contrastive Latent Fields (FCLF)

Implementation of "Learning to Flow by Flowing to Learn" - a novel framework for organizing pretrained embedding spaces by function through learned continuous vector fields.

**Team**: Kyle Kun-Hyung Roh, Vincent Jinpeng Yip
**Course**: CS229 Machine Learning
**Domain**: CelebA face attribute manipulation

## Overview

FCLF learns function-conditioned vector fields in CLIP embedding space to enable controllable image attribute manipulation. The key idea is to organize embeddings by their semantic functions (attributes) through learned flows rather than static latent directions.

### Key Features

- Function-conditioned vector field network
- Contrastive flow loss for attribute-aware organization
- CLIP decoder for visualization
- Support for 5 facial attributes: Smiling, Young, Male, Eyeglasses, Mustache
- Comprehensive evaluation metrics and visualizations

## Project Structure

```
learning-to-flow/
├── data/
│   ├── celeba/                    # CelebA dataset (download separately)
│   └── embeddings/                # Cached CLIP embeddings
├── src/
│   ├── data/                      # Dataset and data utilities
│   ├── models/                    # Vector field and decoder networks
│   ├── losses/                    # Loss functions
│   ├── training/                  # Training scripts
│   ├── evaluation/                # Metrics and visualization
│   └── inference/                 # Inference pipeline
├── scripts/                       # Utility scripts
├── configs/                       # Configuration files
├── checkpoints/                   # Saved model weights
├── results/                       # Outputs and figures
└── notebooks/                     # Jupyter notebooks
```

## Installation

### Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- 32GB RAM (minimum)
- 50GB storage

### Setup

```bash
# Clone repository
git clone <repository-url>
cd learning-to-flow

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Quick Start

### 1. Download Data

```bash
# Follow instructions to download CelebA
./scripts/download_data.sh

# Expected structure:
# data/celeba/
#   ├── img_align_celeba/
#   │   ├── 000001.jpg
#   │   └── ...
#   └── list_attr_celeba.txt
```

### 2. Precompute CLIP Embeddings

```bash
python scripts/precompute_embeddings.py \
    --celeba_root data/celeba \
    --output_dir data/embeddings \
    --batch_size 128
```

This will cache CLIP embeddings for all images (~10-20 minutes on GPU).

### 3. Train Decoder

```bash
./scripts/train_decoder.sh \
    --celeba_root data/celeba \
    --embedding_dir data/embeddings \
    --output_dir outputs/decoder
```

Training takes ~2-3 hours on a single GPU.

### 4. Train FCLF Vector Field

```bash
./scripts/train_fclf.sh \
    --celeba_root data/celeba \
    --embedding_dir data/embeddings \
    --output_dir outputs/fclf
```

Training takes ~4-6 hours on a single GPU.

### 5. Evaluate

```bash
python scripts/run_evaluation.py \
    --checkpoint outputs/fclf/checkpoints/fclf_best.pt \
    --celeba_root data/celeba \
    --embedding_dir data/embeddings \
    --output_dir results/evaluation
```

### 6. Run Inference

```bash
python -m src.inference.inference \
    --image path/to/image.jpg \
    --vector_field outputs/fclf/checkpoints/fclf_best.pt \
    --decoder outputs/decoder/checkpoints/decoder_best.pt \
    --attributes "Smiling:1,Young:1" \
    --output output.png
```

## Configuration

### Decoder Config ([configs/decoder_config.yaml](configs/decoder_config.yaml))

```yaml
model:
  embedding_dim: 512        # CLIP embedding dimension
  img_size: 128            # Output image size

training:
  num_epochs: 20
  batch_size: 64
  learning_rate: 1e-4
```

### FCLF Config ([configs/fclf_config.yaml](configs/fclf_config.yaml))

```yaml
model:
  embedding_dim: 512
  num_attributes: 5        # Number of attributes
  hidden_dim: 256          # Hidden layer size

training:
  num_epochs: 50
  batch_size: 128
  alpha: 0.1              # Flow step size

loss:
  temperature: 0.07        # Contrastive loss temperature
  lambda_curl: 0.01       # Curl regularization weight
  lambda_div: 0.01        # Divergence regularization weight
```

## Usage Examples

### Training with Custom Config

```python
from src.training.train_fclf import train_fclf

train_fclf(
    config_path='configs/fclf_config.yaml',
    celeba_root='data/celeba',
    embedding_dir='data/embeddings',
    output_dir='outputs/fclf_custom',
    use_regularization=True
)
```

### Attribute Manipulation

```python
from src.inference.inference import FCLFInference

# Initialize pipeline
pipeline = FCLFInference(
    vector_field_checkpoint='outputs/fclf/checkpoints/fclf_best.pt',
    decoder_checkpoint='outputs/decoder/checkpoints/decoder_best.pt'
)

# Transform image
transformed = pipeline.transform_image(
    image_path='input.jpg',
    target_attributes={'Smiling': 1, 'Young': 0},
    num_steps=10
)

transformed.save('output.jpg')
```

### Visualization

```python
from src.evaluation.visualize import plot_embedding_2d, plot_trajectory_2d

# Visualize embeddings
plot_embedding_2d(
    embeddings,
    attributes,
    method='umap',
    attribute_names=['Smiling', 'Young', 'Male', 'Eyeglasses', 'Mustache'],
    save_path='embeddings.png'
)

# Visualize trajectories
plot_trajectory_2d(
    trajectories,
    attributes,
    num_samples=20,
    save_path='trajectories.png'
)
```

## Model Architecture

### Vector Field Network

```
Input:
  - z: [batch, 512] CLIP embedding
  - y: [batch, 5] attribute vector

Network:
  - FC(517, 256) + LayerNorm + ReLU
  - FC(256, 256) + LayerNorm + ReLU
  - FC(256, 256) + ReLU
  - FC(256, 512)

Output: v: [batch, 512] velocity vector
```

### CLIP Decoder

```
Input: z: [batch, 512]

Network:
  - FC(512, 512 * 4 * 4)
  - Reshape to [batch, 512, 4, 4]
  - ConvTranspose2d layers: 4→8→16→32→64→128
  - Output: [batch, 3, 128, 128]
```

## Loss Function

```
L_total = L_FCLF + λ_curl * R_curl + λ_div * R_div

Where:
- L_FCLF: InfoNCE contrastive loss on flowed embeddings
- R_curl: ||∇ × v||² (penalize rotational flows)
- R_div: (∇ · v)² (penalize expansion/contraction)
```

## Evaluation Metrics

1. **Silhouette Score**: Measures embedding cluster quality
2. **Cluster Purity**: K-means clustering purity
3. **Attribute Classification**: Accuracy of attribute prediction
4. **Trajectory Smoothness**: L2 distance between consecutive steps

## Success Criteria

- ✓ Pipeline runs end-to-end
- ✓ Decoder MSE < 0.05
- ✓ Silhouette score > 0.2
- ✓ Attribute accuracy > 70%
- ✓ Smooth, interpretable trajectories

## Tips and Troubleshooting

### Training Tips

1. **Start simple**: Train without regularization first (`--no_regularization`)
2. **Monitor losses**: Use TensorBoard to track training
3. **Reduce batch size**: If running out of memory
4. **Use mixed precision**: Add `torch.cuda.amp` for faster training

### Common Issues

**Out of Memory**:
- Reduce batch size in config
- Use gradient accumulation
- Disable regularization (saves memory during backprop)

**Poor Decoder Quality**:
- Train for more epochs
- Try perceptual loss instead of MSE
- Use VAE decoder (`--use_vae`)

**FCLF Not Converging**:
- Reduce learning rate
- Increase flow step size (alpha)
- Add gradient clipping (already included)

## TensorBoard

Monitor training progress:

```bash
tensorboard --logdir outputs/fclf/logs
```

## Citation

If you use this code for your research, please cite:

```
@misc{fclf2024,
  title={Function-Contrastive Latent Fields for Controllable Image Manipulation},
  author={Roh, Kyle Kun-Hyung and Yip, Vincent Jinpeng},
  year={2024},
  course={CS229 Machine Learning}
}
```

## License

MIT License - See LICENSE file for details.

## Contact

- Kyle Kun-Hyung Roh
- Vincent Jinpeng Yip

For questions or issues, please open a GitHub issue.
