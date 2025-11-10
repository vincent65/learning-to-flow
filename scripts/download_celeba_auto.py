#!/usr/bin/env python3
"""
Automated CelebA dataset downloader using Kaggle API.

Prerequisites:
    pip install kaggle

Setup:
    1. Create Kaggle account: https://www.kaggle.com/
    2. Go to Account > API > Create New API Token
    3. This downloads kaggle.json
    4. Place it at ~/.kaggle/kaggle.json (Linux/Mac) or %USERPROFILE%/.kaggle/kaggle.json (Windows)
    5. chmod 600 ~/.kaggle/kaggle.json (Linux/Mac only)

Usage:
    python scripts/download_celeba_auto.py --output_dir data/celeba
"""

import os
import sys
import zipfile
import argparse
from pathlib import Path


def download_celeba_kaggle(output_dir: str):
    """Download CelebA from Kaggle."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CelebA Dataset Downloader (Kaggle)")
    print("=" * 70)

    # Check if kaggle is installed
    try:
        import kaggle
        print("\n✓ Kaggle API found")
    except ImportError:
        print("\n❌ Kaggle API not installed!")
        print("\nPlease install it:")
        print("  pip install kaggle")
        print("\nThen configure your API token:")
        print("  1. Go to https://www.kaggle.com/account")
        print("  2. Click 'Create New API Token'")
        print("  3. Move kaggle.json to ~/.kaggle/")
        print("  4. chmod 600 ~/.kaggle/kaggle.json")
        return False

    # Check if already downloaded
    img_dir = output_dir / "img_align_celeba"
    attr_file = output_dir / "list_attr_celeba.txt"

    if img_dir.exists() and attr_file.exists():
        num_images = len(list(img_dir.glob("*.jpg")))
        if num_images > 200000:
            print(f"\n✓ Dataset already exists ({num_images} images)")
            print(f"  Location: {output_dir}")
            return True

    print(f"\nDownloading CelebA to: {output_dir}")
    print("This will take 10-30 minutes depending on your connection...")

    # Download from Kaggle
    print("\n[1/3] Downloading from Kaggle...")
    try:
        kaggle.api.dataset_download_files(
            'jessicali9530/celeba-dataset',
            path=str(output_dir),
            unzip=True
        )
        print("  ✓ Download complete")
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Check your Kaggle API credentials")
        print("  2. Verify you've accepted the dataset terms on Kaggle")
        print("  3. Try manual download: https://www.kaggle.com/datasets/jessicali9530/celeba-dataset")
        return False

    # Verify download
    print("\n[2/3] Verifying download...")

    if not img_dir.exists():
        print(f"  ⚠ Image directory not found: {img_dir}")
        return False

    num_images = len(list(img_dir.glob("*.jpg")))
    print(f"  ✓ Found {num_images:,} images")

    if not attr_file.exists():
        print(f"  ⚠ Attribute file not found: {attr_file}")
        return False

    print(f"  ✓ Attribute file found")

    # Cleanup
    print("\n[3/3] Cleaning up...")
    # Remove zip files if any
    for zip_file in output_dir.glob("*.zip"):
        try:
            zip_file.unlink()
            print(f"  Removed: {zip_file.name}")
        except:
            pass

    print("\n" + "=" * 70)
    print("✓ CELEBA DATASET READY!")
    print("=" * 70)
    print(f"\nDataset location: {output_dir}")
    print(f"Total images: {num_images:,}")
    print(f"\nYou can now proceed to embedding precomputation:")
    print(f"  python scripts/precompute_embeddings.py --celeba_root {output_dir}")
    print("=" * 70)

    return True


def download_celeba_manual_instructions():
    """Print manual download instructions."""

    print("\n" + "=" * 70)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("=" * 70)
    print("\nSince automated download isn't set up, please download manually:\n")
    print("Option 1: Google Drive (Official)")
    print("-" * 70)
    print("  1. Visit: https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8")
    print("  2. Download: img_align_celeba.zip (1.3GB)")
    print("  3. Download: list_attr_celeba.txt")
    print("  4. Extract img_align_celeba.zip to data/celeba/")
    print("  5. Place list_attr_celeba.txt in data/celeba/")
    print()
    print("Option 2: Kaggle")
    print("-" * 70)
    print("  1. Visit: https://www.kaggle.com/datasets/jessicali9530/celeba-dataset")
    print("  2. Click 'Download' (requires Kaggle account)")
    print("  3. Extract to data/celeba/")
    print()
    print("Expected structure:")
    print("-" * 70)
    print("  data/celeba/")
    print("    ├── img_align_celeba/")
    print("    │   ├── 000001.jpg")
    print("    │   ├── 000002.jpg")
    print("    │   └── ... (202,599 images)")
    print("    └── list_attr_celeba.txt")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download CelebA dataset")
    parser.add_argument("--output_dir", type=str, default="data/celeba",
                        help="Output directory for CelebA dataset")
    parser.add_argument("--manual", action="store_true",
                        help="Show manual download instructions")

    args = parser.parse_args()

    if args.manual:
        download_celeba_manual_instructions()
    else:
        success = download_celeba_kaggle(args.output_dir)

        if not success:
            print("\n" + "=" * 70)
            print("For manual download instructions, run:")
            print("  python scripts/download_celeba_auto.py --manual")
            print("=" * 70)
            sys.exit(1)
