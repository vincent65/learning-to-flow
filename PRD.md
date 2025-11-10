# Product Requirements Document: Function-Contrastive Latent Fields (FCLF)

## Project Overview
Implementation of "Learning to Flow by Flowing to Learn" - a novel framework for organizing pretrained embedding spaces by function through learned continuous vector fields. This project bridges contrastive learning and flow-based models for controllable image attribute manipulation.

**Team**: Kyle Kun-Hyung Roh, Vincent Jinpeng Yip  
**Course**: CS229 Machine Learning  
**Primary Domain**: CelebA face attribute manipulation  
**Timeline**: 4-6 weeks

---

## Core Objectives

### Primary Goals
1. Implement Function-Contrastive Latent Fields (FCLF) framework
2. Train vector field network to learn function-aware flows in CLIP embedding space
3. Enable controllable attribute transformation (e.g., neutral → smiling)
4. Validate approach with quantitative metrics and qualitative visualizations

### Success Criteria
- Vector field successfully organizes embeddings by function (measured by silhouette score)
- Attribute classification accuracy > 80% on flowed embeddings
- Smooth, interpretable latent trajectories
- Computational efficiency: <1 second per inference flow

---

## Technical Architecture

### Component Breakdown

#### 1. Data Pipeline (`src/data/`)
**Purpose**: Load and preprocess CelebA dataset with CLIP embeddings

**Files**:
- `celeba_dataset.py`: CelebA dataset wrapper
- `embedding_cache.py`: Precompute and cache CLIP embeddings
- `data_utils.py`: Preprocessing utilities

**Requirements**:
- Download CelebA dataset (202,599 images)
- Parse `list_attr_celeba.txt` for binary attributes
- Support primary attributes: Smiling, Young, Male, Eyeglasses, Mustache
- Precompute CLIP ViT-B/32 embeddings (512-dim) for all images
- Train/val/test split: 162k/20k/20k (standard CelebA split)
- Efficient batch loading with DataLoader

**Data Format**:
```python
{
    'embedding': torch.Tensor([512]),      # Frozen CLIP embedding
    'attributes': torch.Tensor([5]),       # Binary attribute vector
    'image_id': str,                       # Original filename
    'image_path': str                      # Path to original image (for visualization)
}
```

#### 2. Vector Field Network (`src/models/vector_field.py`)
**Purpose**: Learn function-conditioned vector field v_ω(z, y)

**Architecture**:
```
Input: 
  - z: [batch, 512] CLIP embedding
  - y: [batch, num_attributes] attribute vector

Network:
  - FC(512 + num_attributes, 256) + LayerNorm + ReLU
  - FC(256, 256) + LayerNorm + ReLU  
  - FC(256, 256) + ReLU
  - FC(256, 512)

Output: v: [batch, 512] velocity vector
```

**Key Methods**:
- `forward(z, y)`: Compute velocity field
- `flow(z, y, num_steps, step_size)`: Integrate flow using Euler/RK4
- `get_trajectory(z, y, num_steps)`: Return full trajectory for visualization

#### 3. Loss Functions (`src/losses/`)
**Purpose**: Implement FCLF objective with regularization

**Files**:
- `contrastive_flow_loss.py`: InfoNCE-style contrastive loss on flowed embeddings
- `regularization.py`: Curl and divergence penalties
- `combined_loss.py`: Weighted combination of all losses

**Loss Components**:
```python
L_total = L_FCLF + λ_curl * R_curl + λ_div * R_div

Where:
- L_FCLF: Contrastive loss on z̃ = z + α·v(z, y)
- R_curl: ||∇ × v||² (penalize rotational flows)
- R_div: (∇ · v)² (penalize expansion/contraction)
```

**Hyperparameters**:
- α (flow step size): 0.1 (default)
- τ (temperature): 0.07
- λ_curl: 0.01
- λ_div: 0.01

#### 4. CLIP Decoder (`src/models/clip_decoder.py`)
**Purpose**: Map CLIP embeddings back to images for visualization

**Architecture Options** (implement in order):
1. **Simple Decoder** (MVP): Transposed CNN, L2 reconstruction loss
2. **VAE Decoder** (if time): Add KL divergence, better quality
3. **GAN Decoder** (stretch goal): Adversarial training for photorealism

**Decoder Architecture (Simple)**:
```
Input: z: [batch, 512]

Network:
  - FC(512, 512 * 4 * 4)
  - Reshape to [batch, 512, 4, 4]
  - ConvTranspose2d layers: 4→8→16→32→64→128
  - Output: [batch, 3, 128, 128] image

Loss: MSE(reconstructed, original)
```

**Training Requirements**:
- Separate training phase before FCLF
- Dataset: CelebA images + precomputed CLIP embeddings
- ~10-20 epochs should suffice for reasonable quality
- Save best checkpoint based on validation MSE

#### 5. Training Pipeline (`src/training/`)
**Purpose**: Train FCLF vector field and decoder

**Files**:
- `train_decoder.py`: Pretrain CLIP→image decoder
- `train_fclf.py`: Train vector field network
- `trainer.py`: Unified training loop with logging

**Training Phases**:

**Phase 1: Decoder Pretraining**
```
Input: CLIP embeddings → Output: Reconstructed images
Epochs: 20
Batch size: 64
Optimizer: Adam(lr=1e-4)
Loss: MSE reconstruction
Time estimate: 2-3 hours on single GPU
```

**Phase 2: FCLF Training**
```
Input: CLIP embeddings + attributes
Epochs: 50
Batch size: 128
Optimizer: Adam(lr=1e-4)
Loss: L_FCLF + regularization
Time estimate: 4-6 hours on single GPU
```

**Logging**:
- TensorBoard: loss curves, sample trajectories
- Weights & Biases (optional): hyperparameter tracking
- Checkpoint every 5 epochs
- Validation metrics every epoch

#### 6. Evaluation (`src/evaluation/`)
**Purpose**: Quantitative and qualitative assessment

**Files**:
- `metrics.py`: Silhouette score, cluster purity, attribute accuracy
- `visualize.py`: Generate trajectory plots, reconstructions
- `evaluate.py`: Run full evaluation suite

**Metrics to Implement**:

1. **Silhouette Score**: Measure embedding cluster quality
   - Compute on flowed embeddings grouped by attributes
   - Target: >0.3 (good separation)

2. **Attribute Classification Accuracy**:
   - Train simple classifier: embeddings → attributes
   - Test on flowed embeddings
   - Target: >80% accuracy

3. **Trajectory Smoothness**:
   - Measure L2 distance between consecutive flow steps
   - Lower variance = smoother trajectories

4. **Qualitative Visualizations**:
   - Flow trajectories in 2D (t-SNE/UMAP projection)
   - Image reconstructions at different flow steps
   - Side-by-side comparisons: original → flowed → decoded

#### 7. Inference & Demo (`src/inference/`)
**Purpose**: User-facing interface for attribute manipulation

**Files**:
- `inference.py`: Single-image flow pipeline
- `demo.py`: Gradio web interface (stretch goal)

**Inference Pipeline**:
```python
def transform_image(image_path, target_attributes, num_steps=10):
    1. Load image
    2. Encode with CLIP → z_0
    3. Flow z_0 → z_final using vector field
    4. Decode z_final → image using decoder
    5. Return transformed image + trajectory
```

---

## File Structure

```
fclf-project/
├── data/
│   ├── celeba/                    # CelebA dataset (download separately)
│   │   ├── img_align_celeba/
│   │   └── list_attr_celeba.txt
│   └── embeddings/                # Cached CLIP embeddings
│       ├── train_embeddings.pt
│       ├── val_embeddings.pt
│       └── test_embeddings.pt
│
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── celeba_dataset.py     # Dataset class
│   │   ├── embedding_cache.py    # Precompute CLIP embeddings
│   │   └── data_utils.py         # Utilities
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── vector_field.py       # Vector field network
│   │   ├── clip_decoder.py       # CLIP → image decoder
│   │   └── model_utils.py        # Helper functions
│   │
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── contrastive_flow_loss.py
│   │   ├── regularization.py
│   │   └── combined_loss.py
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train_decoder.py      # Decoder pretraining
│   │   ├── train_fclf.py         # FCLF training
│   │   └── trainer.py            # Unified trainer class
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py            # Evaluation metrics
│   │   ├── visualize.py          # Visualization utilities
│   │   └── evaluate.py           # Evaluation script
│   │
│   └── inference/
│       ├── __init__.py
│       ├── inference.py          # Inference pipeline
│       └── demo.py               # Gradio demo (optional)
│
├── scripts/
│   ├── download_data.sh          # Download CelebA
│   ├── precompute_embeddings.py  # Cache CLIP embeddings
│   ├── train_decoder.sh          # Run decoder training
│   ├── train_fclf.sh             # Run FCLF training
│   └── run_evaluation.py         # Full evaluation
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_decoder_analysis.ipynb
│   ├── 03_fclf_results.ipynb
│   └── 04_ablation_studies.ipynb
│
├── checkpoints/                   # Saved model weights
│   ├── decoder_best.pt
│   └── fclf_best.pt
│
├── results/                       # Outputs
│   ├── figures/
│   ├── metrics/
│   └── generated_images/
│
├── configs/
│   ├── decoder_config.yaml       # Decoder hyperparameters
│   └── fclf_config.yaml          # FCLF hyperparameters
│
├── requirements.txt
├── setup.py
├── README.md
└── CLAUDE.md                      # Detailed implementation guide
```

---

## Implementation Phases

### Phase 1: Data Setup (Week 1)
**Goal**: Prepare dataset and cache embeddings

**Tasks**:
- [ ] Download CelebA dataset
- [ ] Implement `celeba_dataset.py`
- [ ] Precompute CLIP embeddings for all splits
- [ ] Verify data loading pipeline
- [ ] Create data exploration notebook

**Deliverable**: Working data loader with cached embeddings

### Phase 2: Decoder Training (Week 2)
**Goal**: Train CLIP embedding → image decoder

**Tasks**:
- [ ] Implement `clip_decoder.py` (simple transposed CNN)
- [ ] Implement `train_decoder.py`
- [ ] Train decoder on CelebA
- [ ] Evaluate reconstruction quality
- [ ] Save best checkpoint

**Deliverable**: Trained decoder with reasonable reconstruction quality

### Phase 3: FCLF Implementation (Week 3)
**Goal**: Implement vector field and loss functions

**Tasks**:
- [ ] Implement `vector_field.py`
- [ ] Implement contrastive flow loss
- [ ] Implement regularization terms
- [ ] Create `train_fclf.py`
- [ ] Run initial training experiments

**Deliverable**: Working FCLF training pipeline

### Phase 4: Training & Tuning (Week 4)
**Goal**: Train FCLF and tune hyperparameters

**Tasks**:
- [ ] Full FCLF training run
- [ ] Hyperparameter search (α, λ_curl, λ_div)
- [ ] Monitor loss curves and intermediate results
- [ ] Save best checkpoint

**Deliverable**: Trained FCLF model

### Phase 5: Evaluation (Week 5)
**Goal**: Comprehensive evaluation and visualization

**Tasks**:
- [ ] Implement all evaluation metrics
- [ ] Generate qualitative visualizations
- [ ] Create ablation studies
- [ ] Compare to baselines
- [ ] Create results notebooks

**Deliverable**: Complete evaluation results

### Phase 6: Polish & Report (Week 6)
**Goal**: Final touches and documentation

**Tasks**:
- [ ] Create inference demo
- [ ] Write final report
- [ ] Prepare presentation slides
- [ ] Clean up code and documentation

**Deliverable**: Final project submission

---

## Configuration Files

### `configs/decoder_config.yaml`
```yaml
model:
  embedding_dim: 512
  img_size: 128
  hidden_dims: [512, 256, 128, 64, 32]

training:
  num_epochs: 20
  batch_size: 64
  learning_rate: 1e-4
  weight_decay: 1e-5

data:
  train_size: 162000
  val_size: 20000
  num_workers: 4
```

### `configs/fclf_config.yaml`
```yaml
model:
  embedding_dim: 512
  num_attributes: 5
  hidden_dim: 256

training:
  num_epochs: 50
  batch_size: 128
  learning_rate: 1e-4
  alpha: 0.1  # Flow step size
  
loss:
  temperature: 0.07
  lambda_curl: 0.01
  lambda_div: 0.01

inference:
  num_flow_steps: 10
  step_size: 0.1
```

---

## Key Technical Decisions

### 1. CLIP Model Choice
**Decision**: Use CLIP ViT-B/32  
**Rationale**: 
- Good balance of quality and speed
- 512-dim embeddings (manageable size)
- Widely used baseline
- Pretrained weights readily available

### 2. Attributes to Support
**Decision**: Start with 5 attributes (Smiling, Young, Male, Eyeglasses, Mustache)  
**Rationale**:
- Common and well-represented in CelebA
- Mix of facial features and demographics
- Sufficient for proof-of-concept
- Can expand later if needed

### 3. Image Resolution
**Decision**: 128x128 pixels  
**Rationale**:
- Balance between quality and compute
- CelebA images are 178x218, downsampling acceptable
- Faster training than 256x256
- Sufficient for attribute recognition

### 4. Decoder Approach
**Decision**: Start with simple transposed CNN  
**Rationale**:
- Faster to implement and train
- Good enough for validation
- Can upgrade to VAE/GAN if needed
- Reduces project risk

### 5. Regularization Strategy
**Decision**: Use both curl and divergence penalties  
**Rationale**:
- Enforces smooth, conservative flows
- Theoretically motivated
- Prevents degenerate solutions
- Small computational overhead

---

## Risk Mitigation

### Risk 1: Decoder Quality Too Poor
**Impact**: Can't visualize results effectively  
**Mitigation**: 
- Start with low-res (64x64) if needed
- Use perceptual loss if MSE insufficient
- Fallback: nearest-neighbor visualization

### Risk 2: FCLF Training Instability
**Impact**: Vector field doesn't converge  
**Mitigation**:
- Careful learning rate tuning
- Gradient clipping
- Reduce flow step size α
- Add more regularization

### Risk 3: Insufficient Compute Resources
**Impact**: Training takes too long  
**Mitigation**:
- Use smaller subset of CelebA (50k images)
- Reduce batch size
- Use mixed precision training
- Request Farmshare GPU allocation

### Risk 4: Attribute Control Not Working
**Impact**: Flowed embeddings don't change attributes  
**Mitigation**:
- Verify contrastive loss is decreasing
- Check if attributes are learnable (train classifier)
- Increase number of flow steps
- Try different attribute combinations

---

## Success Metrics (Minimum Viable Product)

To consider the project successful, we need:

1. **Functional**: Pipeline runs end-to-end without errors
2. **Decoder**: MSE < 0.05, visual quality acceptable
3. **Clustering**: Silhouette score > 0.2 (some separation)
4. **Accuracy**: Attribute classification > 70% on flowed embeddings
5. **Visualization**: Clear trajectory plots showing movement toward target attributes

**Stretch Goals**:
- Silhouette score > 0.3
- Attribute accuracy > 85%
- Multi-attribute flows (change 2+ attributes simultaneously)
- Gradio demo interface
- Comparison to baseline methods (static latent directions)

---

## Dependencies

### Core Libraries
```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pillow>=9.0.0
```

### CLIP
```
git+https://github.com/openai/CLIP.git
```

### Training & Logging
```
tensorboard>=2.12.0
wandb>=0.15.0  # optional
tqdm>=4.65.0
```

### Evaluation & Visualization
```
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.2.0
umap-learn>=0.5.3
```

### Demo (Optional)
```
gradio>=3.40.0
```

---

## Compute Requirements

### Minimum
- GPU: 1x NVIDIA RTX 2080 Ti (11GB VRAM)
- RAM: 32GB
- Storage: 50GB (dataset + embeddings + checkpoints)
- Time: ~1 week of training

### Recommended
- GPU: 1x NVIDIA A100 (40GB VRAM) or V100
- RAM: 64GB
- Storage: 100GB
- Time: ~2-3 days of training

---

## Baseline Comparisons

For validation, compare FCLF against:

1. **Frozen CLIP**: No transformation, measure baseline clustering
2. **Linear Attribute Directions**: Learn linear vectors in embedding space
3. **Random Flow**: Apply random perturbations instead of learned field

**Comparison Metrics**:
- Silhouette score (clustering quality)
- Attribute classification accuracy
- Smoothness of transitions
- Inference time

---

## Documentation Requirements

### Code Documentation
- Docstrings for all classes and functions
- Type hints for function signatures
- Inline comments for complex logic
- README.md with setup instructions

### Notebooks
- Data exploration and statistics
- Decoder training analysis
- FCLF training results
- Ablation studies and comparisons

### Final Report
- Introduction and motivation
- Related work
- Method description
- Experimental setup
- Results and analysis
- Conclusions and future work

---

## Questions for Clarification

1. **Compute access**: Do you have access to Stanford GPU clusters (Farmshare)?
2. **Timeline flexibility**: Is 6 weeks realistic given other coursework?
3. **Baseline priority**: Should we implement comparison methods or focus on FCLF?
4. **Demo requirement**: Is a Gradio demo expected for the final presentation?

---

## Contact & Collaboration

**Team Communication**:
- Weekly sync meetings
- Shared Google Doc for notes
- GitHub for code collaboration
- Slack/Discord for quick questions

**Responsibilities**:
- **Kyle (KKR)**: Theoretical design, loss functions, protein experiments (if time)
- **Vincent (VJY)**: Implementation, decoder training, visualization, scribe

---

This PRD provides a comprehensive roadmap for implementing FCLF. Adjust phases and priorities based on your timeline and resources. Focus on getting the MVP working first, then iterate on quality and features.
