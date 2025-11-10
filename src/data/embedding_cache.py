"""
Precompute and cache CLIP embeddings for CelebA dataset.
"""

import os
import torch
import clip
from PIL import Image
from tqdm import tqdm
from typing import Optional
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import time
from datetime import datetime


class ImageOnlyDataset(Dataset):
    """Simple dataset that only loads images."""

    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform
        self.error_count = 0
        self.max_errors_to_print = 10

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            # Load image
            image = Image.open(self.image_paths[idx]).convert('RGB')

            # Verify image is valid
            image.load()  # Force load to catch any issues

            # Apply transform
            return self.transform(image)
        except Exception as e:
            self.error_count += 1

            # Only print first few errors to avoid spam
            if self.error_count <= self.max_errors_to_print:
                print(f"\n  ⚠ Error loading image {idx}: {self.image_paths[idx]}")
                print(f"     Error: {e}")
            elif self.error_count == self.max_errors_to_print + 1:
                print(f"\n  ⚠ Suppressing further error messages (total errors: {self.error_count})")

            # Return a blank tensor with correct shape for CLIP
            # CLIP preprocessing expects 224x224
            try:
                # Try to create a blank image and process it normally
                blank_image = Image.new('RGB', (224, 224), color=(128, 128, 128))
                return self.transform(blank_image)
            except:
                # Ultimate fallback: return properly shaped tensor
                return torch.zeros(3, 224, 224)


def precompute_embeddings(
    celeba_root: str,
    output_dir: str,
    clip_model_name: str = 'ViT-B/32',
    batch_size: int = 128,
    num_workers: int = 4,
    device: Optional[str] = None
):
    """
    Precompute CLIP embeddings for all CelebA images and save to disk.

    Args:
        celeba_root: Path to CelebA root directory
        output_dir: Directory to save embeddings
        clip_model_name: CLIP model to use
        batch_size: Batch size for encoding
        num_workers: Number of data loading workers
        device: Device to use (defaults to cuda if available)
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 70)
    print(f"CLIP Embedding Precomputation Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print(f"\n[1/6] Configuration:")
    print(f"  Device: {device}")
    print(f"  CLIP Model: {clip_model_name}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Num Workers: {num_workers}")
    print(f"  CelebA Root: {celeba_root}")
    print(f"  Output Dir: {output_dir}")

    # Load CLIP model
    print(f"\n[2/6] Loading CLIP model...")
    start_time = time.time()
    model, preprocess = clip.load(clip_model_name, device=device)
    model.eval()
    print(f"  ✓ Model loaded in {time.time() - start_time:.2f} seconds")

    # Get all image paths
    print(f"\n[3/6] Loading image paths...")
    img_dir = os.path.join(celeba_root, 'img_align_celeba')
    attr_file = os.path.join(celeba_root, 'list_attr_celeba.txt')

    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    if not os.path.exists(attr_file):
        raise FileNotFoundError(f"Attribute file not found: {attr_file}")

    # Read image names from attribute file to preserve order
    with open(attr_file, 'r') as f:
        lines = f.readlines()[2:]  # Skip header

    image_names = [line.split()[0] for line in lines]
    image_paths = [os.path.join(img_dir, name) for name in image_names]

    total_images = len(image_paths)
    print(f"  ✓ Found {total_images:,} images")

    # Verify a few images exist
    print(f"  Verifying image files exist...")
    missing_count = 0
    for i in range(min(100, len(image_paths))):
        if not os.path.exists(image_paths[i]):
            missing_count += 1
    if missing_count > 0:
        print(f"  ⚠ Warning: {missing_count}/100 sample images missing")
    else:
        print(f"  ✓ Sample verification passed")

    # Create dataset and dataloader
    print(f"\n[4/6] Creating data loader...")
    dataset = ImageOnlyDataset(image_paths, preprocess)

    # Note: If you get "Numpy is not available" errors, try running with --num_workers 0
    # This disables multiprocessing which can have compatibility issues
    if num_workers > 0:
        print(f"  Note: Using {num_workers} workers. If you see errors, try --num_workers 0")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device == 'cuda' else False
    )
    num_batches = len(dataloader)
    print(f"  ✓ DataLoader ready ({num_batches:,} batches of size {batch_size})")

    # Compute embeddings
    all_embeddings = []

    print(f"\n[5/6] Computing embeddings...")
    print(f"  This will take approximately {num_batches * 0.5 / 60:.1f}-{num_batches * 1.0 / 60:.1f} minutes")
    print(f"  Progress will be shown below:")
    print()

    start_encode_time = time.time()
    batch_times = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="  Encoding images", ncols=80)):
            batch_start = time.time()

            batch = batch.to(device)
            embeddings = model.encode_image(batch)
            # Normalize embeddings
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            all_embeddings.append(embeddings.cpu())

            batch_time = time.time() - batch_start
            batch_times.append(batch_time)

            # Print periodic updates every 100 batches
            if (batch_idx + 1) % 100 == 0:
                avg_batch_time = sum(batch_times[-100:]) / len(batch_times[-100:])
                remaining_batches = num_batches - (batch_idx + 1)
                eta_seconds = remaining_batches * avg_batch_time
                eta_minutes = eta_seconds / 60

                images_processed = (batch_idx + 1) * batch_size
                percent_done = (batch_idx + 1) / num_batches * 100

                print(f"\n  Progress: {images_processed:,}/{total_images:,} images ({percent_done:.1f}%)")
                print(f"  Speed: {avg_batch_time:.3f}s/batch, ETA: {eta_minutes:.1f} min")

    encoding_time = time.time() - start_encode_time
    print(f"\n  ✓ Encoding complete in {encoding_time / 60:.2f} minutes")
    print(f"  Average speed: {total_images / encoding_time:.1f} images/second")

    # Report any errors
    if dataset.error_count > 0:
        print(f"\n  ⚠ Warning: {dataset.error_count} images failed to load")
        print(f"     These were replaced with blank embeddings")
        print(f"     Success rate: {(total_images - dataset.error_count) / total_images * 100:.2f}%")

    # Concatenate all embeddings
    print(f"\n  Concatenating embeddings...")
    all_embeddings = torch.cat(all_embeddings, dim=0)
    print(f"  ✓ Final embeddings shape: {all_embeddings.shape}")
    print(f"  Memory usage: {all_embeddings.element_size() * all_embeddings.nelement() / 1024 / 1024:.1f} MB")

    # Create output directory
    print(f"\n[6/6] Saving embeddings to disk...")
    os.makedirs(output_dir, exist_ok=True)

    # Split into train/val/test following standard CelebA splits
    train_end = 162000
    val_end = 182000

    train_embeddings = all_embeddings[:train_end]
    val_embeddings = all_embeddings[train_end:val_end]
    test_embeddings = all_embeddings[val_end:]

    # Save embeddings
    train_path = os.path.join(output_dir, 'train_embeddings.pt')
    val_path = os.path.join(output_dir, 'val_embeddings.pt')
    test_path = os.path.join(output_dir, 'test_embeddings.pt')

    print(f"\n  Saving train split ({train_embeddings.shape[0]:,} samples)...")
    save_start = time.time()
    torch.save(train_embeddings, train_path)
    train_size_mb = os.path.getsize(train_path) / 1024 / 1024
    print(f"  ✓ Saved to {train_path} ({train_size_mb:.1f} MB) in {time.time() - save_start:.2f}s")

    print(f"\n  Saving validation split ({val_embeddings.shape[0]:,} samples)...")
    save_start = time.time()
    torch.save(val_embeddings, val_path)
    val_size_mb = os.path.getsize(val_path) / 1024 / 1024
    print(f"  ✓ Saved to {val_path} ({val_size_mb:.1f} MB) in {time.time() - save_start:.2f}s")

    print(f"\n  Saving test split ({test_embeddings.shape[0]:,} samples)...")
    save_start = time.time()
    torch.save(test_embeddings, test_path)
    test_size_mb = os.path.getsize(test_path) / 1024 / 1024
    print(f"  ✓ Saved to {test_path} ({test_size_mb:.1f} MB) in {time.time() - save_start:.2f}s")

    total_time = time.time() - start_encode_time
    total_size_mb = train_size_mb + val_size_mb + test_size_mb

    print("\n" + "=" * 70)
    print("✓ EMBEDDING PRECOMPUTATION COMPLETE!")
    print("=" * 70)
    print(f"\nSummary:")
    print(f"  Total images processed: {total_images:,}")
    print(f"  Total time: {total_time / 60:.2f} minutes")
    print(f"  Total disk space: {total_size_mb:.1f} MB")
    print(f"\nOutput files:")
    print(f"  Train: {train_path}")
    print(f"  Val:   {val_path}")
    print(f"  Test:  {test_path}")
    print(f"\nYou can now proceed to training!")
    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Precompute CLIP embeddings for CelebA")
    parser.add_argument("--celeba_root", type=str, required=True,
                        help="Path to CelebA root directory")
    parser.add_argument("--output_dir", type=str, default="data/embeddings",
                        help="Output directory for embeddings")
    parser.add_argument("--clip_model", type=str, default="ViT-B/32",
                        help="CLIP model name")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu)")

    args = parser.parse_args()

    precompute_embeddings(
        celeba_root=args.celeba_root,
        output_dir=args.output_dir,
        clip_model_name=args.clip_model,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device
    )
