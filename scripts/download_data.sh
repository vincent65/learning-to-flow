#!/bin/bash
# Download CelebA dataset

# This script provides instructions for downloading CelebA
# CelebA requires manual download from official sources

echo "CelebA Dataset Download Instructions"
echo "====================================="
echo ""
echo "CelebA dataset requires manual download from one of these sources:"
echo ""
echo "Option 1: Google Drive (Official)"
echo "  1. Visit: https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8"
echo "  2. Download: img_align_celeba.zip (1.3GB)"
echo "  3. Download: list_attr_celeba.txt"
echo ""
echo "Option 2: Kaggle"
echo "  1. Visit: https://www.kaggle.com/datasets/jessicali9530/celeba-dataset"
echo "  2. Download the dataset"
echo ""
echo "After downloading:"
echo "  1. Extract img_align_celeba.zip to data/celeba/img_align_celeba/"
echo "  2. Place list_attr_celeba.txt in data/celeba/"
echo ""
echo "Expected structure:"
echo "  data/celeba/"
echo "    ├── img_align_celeba/"
echo "    │   ├── 000001.jpg"
echo "    │   ├── 000002.jpg"
echo "    │   └── ..."
echo "    └── list_attr_celeba.txt"
echo ""
echo "Total images: 202,599"
echo "Total size: ~1.4GB"
