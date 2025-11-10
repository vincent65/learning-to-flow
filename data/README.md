# Data Directory

This directory contains the CelebA dataset and cached CLIP embeddings.

## Directory Structure

```
data/
├── celeba/                     # CelebA dataset (not in Git)
│   ├── img_align_celeba/       # 202,599 face images
│   └── list_attr_celeba.txt    # Attribute labels
└── embeddings/                 # Cached CLIP embeddings (not in Git)
    ├── train_embeddings.pt     # 162,000 training embeddings
    ├── val_embeddings.pt       # 20,000 validation embeddings
    └── test_embeddings.pt      # 20,599 test embeddings
```

## Why This Directory Is Not in Git

- **Dataset is too large**: 1.7 GB (exceeds GitHub limits)
- **Best practice**: Datasets should not be version controlled
- **Licensing**: CelebA has specific usage terms
- **Reproducibility**: Anyone can download the same dataset

## How to Download CelebA

### Option 1: Automated Download (Recommended)

```bash
# Install Kaggle API
pip install kaggle

# Configure Kaggle credentials (first time only)
# 1. Go to https://www.kaggle.com/account
# 2. Click "Create New API Token"
# 3. Move kaggle.json to ~/.kaggle/
# 4. chmod 600 ~/.kaggle/kaggle.json

# Download dataset
python scripts/download_celeba_auto.py --output_dir data/celeba
```

### Option 2: Manual Download

**From Google Drive (Official):**
1. Visit: https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8
2. Download: `img_align_celeba.zip` (1.3GB)
3. Download: `list_attr_celeba.txt`
4. Extract to `data/celeba/`

**From Kaggle:**
1. Visit: https://www.kaggle.com/datasets/jessicali9530/celeba-dataset
2. Click "Download" (requires account)
3. Extract to `data/celeba/`

### Verify Download

```bash
# Check structure
ls -la data/celeba/

# Should show:
# img_align_celeba/       (directory with 202,599 .jpg files)
# list_attr_celeba.txt    (text file)

# Count images
ls data/celeba/img_align_celeba/*.jpg | wc -l
# Should output: 202599
```

## Generate CLIP Embeddings

After downloading CelebA, precompute embeddings:

```bash
python scripts/precompute_embeddings.py \
    --celeba_root data/celeba \
    --output_dir data/embeddings \
    --batch_size 128 \
    --num_workers 0
```

This takes 20-30 minutes and creates ~400 MB of embedding files.

## Dataset Information

### CelebA Dataset
- **Full name**: CelebFaces Attributes Dataset
- **Images**: 202,599 face images
- **Resolution**: 178×218 pixels (aligned and cropped)
- **Attributes**: 40 binary attributes per image
- **Splits**: 162,770 train / 19,867 val / 19,962 test
- **Paper**: Liu et al., "Deep Learning Face Attributes in the Wild", ICCV 2015
- **License**: Non-commercial research purposes only

### Our Usage
We use 5 primary attributes:
- Smiling
- Young
- Male
- Eyeglasses
- Mustache

### CLIP Embeddings
- **Model**: CLIP ViT-B/32
- **Dimension**: 512
- **Normalization**: L2 normalized
- **Format**: PyTorch tensors (.pt files)
- **Size**: ~400 MB total

## Citation

If you use CelebA in your research, please cite:

```bibtex
@inproceedings{liu2015faceattributes,
  title = {Deep Learning Face Attributes in the Wild},
  author = {Liu, Ziwei and Luo, Ping and Wang, Xiaogang and Tang, Xiaoou},
  booktitle = {Proceedings of International Conference on Computer Vision (ICCV)},
  month = {December},
  year = {2015}
}
```

## Troubleshooting

**Problem**: Download fails
- Check internet connection
- Verify Kaggle credentials
- Try manual download instead

**Problem**: Wrong directory structure
- Make sure files are in `data/celeba/` not `data/celeba/celeba/`
- Check that `img_align_celeba/` is a direct subdirectory

**Problem**: Missing images
- Verify you have 202,599 .jpg files
- Re-download if count is wrong

For more help, see [TROUBLESHOOTING.md](../TROUBLESHOOTING.md)
