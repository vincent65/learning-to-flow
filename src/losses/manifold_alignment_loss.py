"""
Manifold Alignment Loss

Forces flowed embeddings to stay close to actual training embeddings.
This prevents flowing to "phantom" regions off the manifold.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ManifoldAlignmentLoss(nn.Module):
    """
    Ensures flowed embeddings stay close to real training embeddings.
    """

    def __init__(
        self,
        train_embeddings: torch.Tensor,
        train_attributes: torch.Tensor,
        k_neighbors: int = 5,
        temperature: float = 0.1
    ):
        """
        Args:
            train_embeddings: [N_train, dim] all training embeddings
            train_attributes: [N_train, num_attrs] all training attributes
            k_neighbors: How many nearest neighbors to consider
            temperature: Temperature for soft matching
        """
        super().__init__()
        self.register_buffer('train_embeddings', train_embeddings)
        self.register_buffer('train_attributes', train_attributes)
        self.k_neighbors = k_neighbors
        self.temperature = temperature

    def forward(
        self,
        z_flowed: torch.Tensor,
        target_attrs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute manifold alignment loss.

        For each flowed embedding, find the k nearest training embeddings
        with matching target attributes, and pull toward them.

        Args:
            z_flowed: [batch, dim] flowed embeddings
            target_attrs: [batch, num_attrs] target attributes

        Returns:
            loss: scalar
        """
        batch_size = z_flowed.size(0)

        # Normalize for cosine similarity
        z_flowed_norm = F.normalize(z_flowed, dim=1)
        train_norm = F.normalize(self.train_embeddings, dim=1)

        # For each batch sample
        total_loss = 0.0
        for i in range(batch_size):
            target = target_attrs[i]  # [num_attrs]

            # Find training samples with matching attributes
            # Allow partial matches (at least 4/5 attributes match)
            attr_match = (self.train_attributes == target.unsqueeze(0)).float()
            match_score = attr_match.sum(dim=1)  # [N_train]
            threshold = target.size(0) - 1  # Allow 1 mismatch

            mask = (match_score >= threshold).float()  # [N_train]

            if mask.sum() < self.k_neighbors:
                # Not enough matching samples, skip
                continue

            # Compute cosine similarities to all training embeddings
            similarities = torch.matmul(train_norm, z_flowed_norm[i])  # [N_train]

            # Mask to only matching attributes
            masked_sim = similarities * mask

            # Get k nearest with matching attributes
            top_k_values, top_k_indices = torch.topk(masked_sim, k=self.k_neighbors)

            # Pull toward these neighbors (softly)
            # distance = 1 - cosine_sim
            distances = 1 - top_k_values
            loss = distances.mean()

            total_loss = total_loss + loss

        return total_loss / batch_size


def create_manifold_loss(
    train_loader,
    device: str = 'cuda'
) -> ManifoldAlignmentLoss:
    """
    Helper to create manifold loss from training data.

    Args:
        train_loader: DataLoader for training set
        device: 'cuda' or 'cpu'

    Returns:
        ManifoldAlignmentLoss module
    """
    print("Loading training embeddings for manifold alignment loss...")
    all_embeddings = []
    all_attributes = []

    for batch in train_loader:
        all_embeddings.append(batch['embedding'])
        all_attributes.append(batch['attributes'])

    train_embeddings = torch.cat(all_embeddings).to(device)
    train_attributes = torch.cat(all_attributes).to(device)

    print(f"  Loaded {len(train_embeddings)} training samples")

    return ManifoldAlignmentLoss(
        train_embeddings,
        train_attributes,
        k_neighbors=5,
        temperature=0.1
    )
