#!/usr/bin/env python3
"""
Script to precompute CLIP embeddings for CelebA dataset.

This should be run once before training to cache all embeddings.
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.embedding_cache import precompute_embeddings
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute CLIP embeddings for CelebA")
    parser.add_argument("--celeba_root", type=str, default="data/celeba",
                        help="Path to CelebA root directory")
    parser.add_argument("--output_dir", type=str, default="data/embeddings",
                        help="Output directory for embeddings")
    parser.add_argument("--clip_model", type=str, default="ViT-B/32",
                        help="CLIP model name")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for encoding")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu)")

    args = parser.parse_args()

    print("=" * 60)
    print("Precomputing CLIP Embeddings for CelebA")
    print("=" * 60)
    print(f"CelebA root: {args.celeba_root}")
    print(f"Output dir: {args.output_dir}")
    print(f"CLIP model: {args.clip_model}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 60)

    precompute_embeddings(
        celeba_root=args.celeba_root,
        output_dir=args.output_dir,
        clip_model_name=args.clip_model,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device
    )

    print("\n" + "=" * 60)
    print("Embedding precomputation complete!")
    print("=" * 60)
