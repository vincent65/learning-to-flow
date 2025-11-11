"""
CelebA Dataset wrapper for FCLF project.
Loads precomputed CLIP embeddings and attribute labels.
"""

import os
from typing import Dict, Optional, Tuple
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class CelebADataset(Dataset):
    """CelebA dataset with CLIP embeddings and attributes."""

    # Primary attributes to use
    ATTRIBUTES = ['Smiling', 'Young', 'Male', 'Eyeglasses', 'Mustache']

    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        embedding_path: Optional[str] = None,
        load_images: bool = False,
        transform=None
    ):
        """
        Args:
            root_dir: Path to CelebA root directory containing img_align_celeba/ and list_attr_celeba.txt
            split: One of 'train', 'val', 'test'
            embedding_path: Path to precomputed embeddings (.pt file)
            load_images: Whether to load actual images (for visualization/decoder training)
            transform: Optional transform to apply to images
        """
        self.root_dir = root_dir
        self.split = split
        self.load_images = load_images
        self.transform = transform

        # Load attribute labels
        self.attributes_df = self._load_attributes()

        # Load embeddings if path provided
        if embedding_path:
            self.embeddings = torch.load(embedding_path)
        else:
            self.embeddings = None

        # Get image paths
        self.img_dir = os.path.join(root_dir, 'img_align_celeba')

        # Standard CelebA splits
        self.indices = self._get_split_indices()

    def _load_attributes(self) -> Dict[str, torch.Tensor]:
        """Load and parse attribute file."""
        attr_file = os.path.join(self.root_dir, 'list_attr_celeba.txt')

        if not os.path.exists(attr_file):
            raise FileNotFoundError(f"Attribute file not found: {attr_file}")

        # Read file
        with open(attr_file, 'r') as f:
            lines = f.readlines()

        # First line is count, second line is header
        num_images = int(lines[0].strip())
        header = lines[1].strip().split()

        # Find indices of our target attributes
        attr_indices = [header.index(attr) for attr in self.ATTRIBUTES]

        # Parse data
        image_names = []
        attributes = []

        for line in lines[2:]:
            parts = line.strip().split()
            image_names.append(parts[0])

            # Extract target attributes (convert -1/1 to 0/1)
            attr_values = [(int(parts[i+1]) + 1) // 2 for i in attr_indices]
            attributes.append(attr_values)

        return {
            'image_names': image_names,
            'attributes': torch.tensor(attributes, dtype=torch.float32)
        }

    def _get_split_indices(self) -> range:
        """Get indices for train/val/test split following standard CelebA splits."""
        total = len(self.attributes_df['image_names'])

        # Standard CelebA splits
        train_end = 162000
        val_end = 182000

        if self.split == 'train':
            return range(0, train_end)
        elif self.split == 'val':
            return range(train_end, val_end)
        elif self.split == 'test':
            return range(val_end, total)
        else:
            raise ValueError(f"Invalid split: {self.split}. Must be 'train', 'val', or 'test'")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dictionary containing:
                - embedding: [512] CLIP embedding (if available)
                - attributes: [5] binary attribute vector
                - image_id: filename
                - image_path: full path to image
                - image: [3, 128, 128] image tensor (if load_images=True)
        """
        actual_idx = self.indices[idx]

        # Get image name and path
        image_name = self.attributes_df['image_names'][actual_idx]
        image_path = os.path.join(self.img_dir, image_name)

        # Get attributes
        attributes = self.attributes_df['attributes'][actual_idx]

        # Build return dict
        item = {
            'attributes': attributes,
            'image_id': image_name,
            'image_path': image_path
        }

        # Add embedding if available
        # Note: use idx (not actual_idx) because embeddings are split-specific
        # idx is 0-based within this split (e.g., 0-19999 for val)
        # actual_idx is global (e.g., 162000-181999 for val)
        if self.embeddings is not None:
            item['embedding'] = self.embeddings[idx]

        # Load image if requested
        if self.load_images:
            try:
                image = Image.open(image_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                item['image'] = image
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                # Return black image as fallback
                item['image'] = torch.zeros(3, 128, 128)

        return item


def get_dataloader(
    root_dir: str,
    split: str,
    batch_size: int,
    embedding_path: Optional[str] = None,
    load_images: bool = False,
    transform=None,
    num_workers: int = 4,
    shuffle: bool = None
) -> torch.utils.data.DataLoader:
    """
    Convenience function to create a DataLoader.

    Args:
        root_dir: Path to CelebA dataset
        split: 'train', 'val', or 'test'
        batch_size: Batch size
        embedding_path: Path to precomputed embeddings
        load_images: Whether to load images
        transform: Image transforms
        num_workers: Number of workers for data loading
        shuffle: Whether to shuffle (defaults to True for train, False otherwise)
    """
    if shuffle is None:
        shuffle = (split == 'train')

    dataset = CelebADataset(
        root_dir=root_dir,
        split=split,
        embedding_path=embedding_path,
        load_images=load_images,
        transform=transform
    )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
