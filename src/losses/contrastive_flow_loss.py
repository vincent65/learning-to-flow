"""
Contrastive Flow Loss (InfoNCE-style) for FCLF.

The key idea: embeddings flowed toward the same target attributes should be similar,
while embeddings with different target attributes should be dissimilar.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveFlowLoss(nn.Module):
    """
    InfoNCE-style contrastive loss on flowed embeddings.

    For each embedding, positive samples are those with the same target attributes,
    and negative samples are those with different target attributes.
    """

    def __init__(self, temperature: float = 0.07):
        """
        Args:
            temperature: Temperature parameter for softmax
        """
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        z_flowed: torch.Tensor,
        attributes: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss on flowed embeddings.

        Args:
            z_flowed: [batch, embedding_dim] flowed embeddings
            attributes: [batch, num_attributes] target attribute vectors

        Returns:
            loss: scalar contrastive loss
        """
        batch_size = z_flowed.size(0)

        # Normalize embeddings for cosine similarity
        z_norm = F.normalize(z_flowed, dim=1)

        # Compute similarity matrix: [batch, batch]
        similarity = torch.matmul(z_norm, z_norm.t()) / self.temperature

        # Create mask for positive pairs (same attributes)
        # Two samples are positive if they have identical attribute vectors
        attr_similarity = torch.matmul(attributes, attributes.t())
        num_attrs = attributes.size(1)

        # Positive mask: all attributes match
        positive_mask = (attr_similarity == num_attrs).float()

        # Remove self-similarity
        positive_mask.fill_diagonal_(0)

        # Negative mask: at least one attribute differs
        negative_mask = (attr_similarity < num_attrs).float()

        # Check if we have valid positive pairs
        num_positives = positive_mask.sum(dim=1)

        if num_positives.sum() == 0:
            # No positive pairs, return zero loss
            return torch.tensor(0.0, device=z_flowed.device, requires_grad=True)

        # Compute InfoNCE loss
        # For each anchor, maximize similarity to positives, minimize to negatives
        losses = []

        for i in range(batch_size):
            if num_positives[i] == 0:
                continue

            # Similarities for this anchor
            pos_sims = similarity[i] * positive_mask[i]
            neg_sims = similarity[i] * negative_mask[i]

            # Log-sum-exp for numerator (positives)
            pos_exp = torch.exp(pos_sims)
            pos_sum = pos_exp.sum()

            # Log-sum-exp for denominator (all except self)
            all_exp = torch.exp(similarity[i])
            all_sum = all_exp.sum() - all_exp[i]  # Exclude self

            # Loss for this anchor
            if all_sum > 0 and pos_sum > 0:
                loss_i = -torch.log(pos_sum / all_sum)
                losses.append(loss_i)

        if len(losses) == 0:
            return torch.tensor(0.0, device=z_flowed.device, requires_grad=True)

        # Average over valid anchors
        loss = torch.stack(losses).mean()

        return loss


class AttributeContrastiveLoss(nn.Module):
    """
    Improved contrastive loss with prototype-based clustering.

    Key improvements over previous version:
    1. Computes attribute prototypes (centroids) for each attribute combination
    2. Pulls flowed embeddings toward their target attribute prototype
    3. Pushes embeddings away from other attribute prototypes

    This creates explicit attribute-specific clusters and prevents mode collapse.
    """

    def __init__(self, temperature: float = 0.07):
        """
        Args:
            temperature: Temperature parameter for contrastive loss
        """
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        z_flowed: torch.Tensor,
        attributes: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute prototype-based contrastive loss.

        Args:
            z_flowed: [batch, embedding_dim] flowed embeddings
            attributes: [batch, num_attributes] target attribute vectors

        Returns:
            loss: scalar contrastive loss
        """
        batch_size = z_flowed.size(0)
        num_attrs = attributes.size(1)

        # Normalize embeddings
        z_norm = F.normalize(z_flowed, dim=1)

        # Compute attribute prototypes (centroids for each unique attribute combination)
        # This creates explicit targets for the model to flow toward
        unique_attrs, inverse_indices = torch.unique(
            attributes, dim=0, return_inverse=True
        )

        num_prototypes = unique_attrs.size(0)

        # Compute prototype embeddings as mean of embeddings with same attributes
        # CRITICAL: Use stop-gradient (.detach()) to prevent prototype collapse!
        # Without detach, prototypes and embeddings collapse together to arbitrary points.
        # With detach, prototypes act as semi-fixed targets computed from current batch.
        prototype_list = []

        for i in range(num_prototypes):
            mask = (inverse_indices == i)
            if mask.sum() > 0:
                # Compute mean embedding for this attribute combination
                proto = z_norm[mask].mean(dim=0, keepdim=True)
                # Normalize prototype (non-in-place)
                proto = F.normalize(proto, dim=1)
                # CRITICAL: Detach from gradient graph to prevent collapse
                proto = proto.detach()
                prototype_list.append(proto)
            else:
                # Empty prototype (shouldn't happen but handle gracefully)
                proto = torch.zeros(1, z_norm.size(1), device=z_norm.device, dtype=z_norm.dtype)
                prototype_list.append(proto)

        # Stack prototypes [num_prototypes, embedding_dim]
        prototypes = torch.cat(prototype_list, dim=0)

        # Compute similarity between each embedding and all prototypes
        # [batch, num_prototypes]
        proto_similarities = torch.matmul(z_norm, prototypes.t()) / self.temperature

        # Create target: each embedding should match its own prototype
        # [batch, num_prototypes] one-hot encoding
        targets = torch.zeros_like(proto_similarities)
        targets[torch.arange(batch_size), inverse_indices] = 1.0

        # InfoNCE-style loss: maximize similarity to correct prototype,
        # minimize similarity to other prototypes
        exp_similarities = torch.exp(proto_similarities)

        # Numerator: similarity to correct prototype
        pos_similarities = (exp_similarities * targets).sum(dim=1)

        # Denominator: sum of all similarities
        all_similarities = exp_similarities.sum(dim=1)

        # Avoid division by zero
        all_similarities = torch.clamp(all_similarities, min=1e-8)

        # Cross-entropy loss
        loss = -torch.log(pos_similarities / all_similarities + 1e-8).mean()

        # Additional pairwise contrastive term for within-batch clustering
        # This helps when prototypes are under-sampled
        pairwise_sim = torch.matmul(z_norm, z_norm.t()) / self.temperature

        # Compute attribute similarity matrix
        attr_similarity = torch.matmul(attributes, attributes.t())  # [batch, batch]

        # Positive pairs: same attributes
        positive_mask = (attr_similarity == num_attrs).float()
        positive_mask.fill_diagonal_(0)  # Exclude self

        # Negative pairs: different attributes
        negative_mask = (attr_similarity < num_attrs).float()

        # Compute pairwise contrastive loss
        if positive_mask.sum() > 0:
            # For each sample, compute InfoNCE over its positives vs negatives
            exp_pairwise = torch.exp(pairwise_sim)

            pos_sims = (exp_pairwise * positive_mask).sum(dim=1)
            all_sims = exp_pairwise.sum(dim=1) - exp_pairwise.diagonal()  # Exclude self

            # Only compute for samples that have positive pairs
            valid_mask = (positive_mask.sum(dim=1) > 0)
            if valid_mask.sum() > 0:
                pairwise_loss = -torch.log(
                    pos_sims[valid_mask] / (all_sims[valid_mask] + 1e-8) + 1e-8
                ).mean()

                # Combine prototype loss and pairwise loss
                loss = 0.7 * loss + 0.3 * pairwise_loss

        return loss


class SimpleContrastiveLoss(nn.Module):
    """
    Simplified contrastive loss using one-step flow.

    L_FCLF = -log(exp(sim(z̃, z_pos)) / Σ exp(sim(z̃, z_neg)))

    where z̃ = z + α·v(z, y)
    """

    def __init__(self, temperature: float = 0.07, alpha: float = 0.1):
        """
        Args:
            temperature: Temperature for softmax
            alpha: Flow step size
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(
        self,
        z_original: torch.Tensor,
        z_flowed: torch.Tensor,
        attributes: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss.

        Args:
            z_original: [batch, dim] original embeddings
            z_flowed: [batch, dim] flowed embeddings (z + α·v)
            attributes: [batch, num_attrs] target attributes

        Returns:
            loss: scalar
        """
        # Normalize
        z_flowed_norm = F.normalize(z_flowed, dim=1)

        # Compute pairwise similarities
        sim_matrix = torch.matmul(z_flowed_norm, z_flowed_norm.t()) / self.temperature

        # Create labels: same attributes = positive
        attr_sim = torch.matmul(attributes, attributes.t())
        num_attrs = attributes.size(1)
        labels = (attr_sim == num_attrs).long()

        # Remove diagonal
        labels.fill_diagonal_(0)

        # Compute cross-entropy style loss
        loss = 0
        count = 0

        for i in range(z_flowed.size(0)):
            pos_mask = labels[i].bool()
            if pos_mask.sum() == 0:
                continue

            # Positive and negative similarities
            pos_sim = sim_matrix[i][pos_mask]
            neg_mask = ~pos_mask
            neg_mask[i] = False  # Exclude self
            neg_sim = sim_matrix[i][neg_mask]

            # InfoNCE
            pos_exp = torch.exp(pos_sim).sum()
            all_exp = torch.exp(torch.cat([pos_sim, neg_sim])).sum()

            if all_exp > 0:
                loss += -torch.log(pos_exp / all_exp)
                count += 1

        return loss / count if count > 0 else torch.tensor(0.0, device=z_flowed.device)
