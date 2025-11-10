"""
Data preprocessing utilities for FCLF project.
"""

import torch
import torchvision.transforms as transforms


def get_image_transforms(img_size: int = 128, is_train: bool = True):
    """
    Get image transforms for training or evaluation.

    Args:
        img_size: Target image size
        is_train: Whether this is for training (adds augmentation)

    Returns:
        torchvision transforms
    """
    if is_train:
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


def denormalize_image(tensor: torch.Tensor) -> torch.Tensor:
    """
    Denormalize image tensor for visualization.

    Args:
        tensor: Normalized image tensor [3, H, W] or [B, 3, H, W]

    Returns:
        Denormalized tensor in [0, 1] range
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)

    if tensor.dim() == 4:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)

    denormalized = tensor * std + mean
    return torch.clamp(denormalized, 0, 1)


def get_attribute_stats(dataset):
    """
    Compute statistics about attribute distribution in dataset.

    Args:
        dataset: CelebADataset instance

    Returns:
        Dictionary with attribute statistics
    """
    all_attrs = []
    for i in range(len(dataset)):
        all_attrs.append(dataset[i]['attributes'])

    all_attrs = torch.stack(all_attrs)
    positive_counts = all_attrs.sum(dim=0)
    total = len(dataset)

    stats = {}
    for i, attr_name in enumerate(dataset.ATTRIBUTES):
        count = positive_counts[i].item()
        stats[attr_name] = {
            'positive': int(count),
            'negative': int(total - count),
            'ratio': count / total
        }

    return stats
