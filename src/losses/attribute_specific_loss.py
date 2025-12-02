"""
Loss function for attribute-specific vector fields.

Key differences from original:
1. Per-attribute contrastive loss (not all-or-nothing)
2. Orthogonality loss (attribute flows should be independent)
3. Cycle consistency (flow forward then back should return to start)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttributeSpecificLoss(nn.Module):
    """
    Loss for training attribute-specific vector fields.
    """

    def __init__(
        self,
        temperature: float = 0.2,
        lambda_contrastive: float = 0.5,
        lambda_identity: float = 0.1,
        lambda_curl: float = 0.01,
        lambda_div: float = 0.01,
        lambda_orthogonal: float = 0.1,
        lambda_cycle: float = 0.1
    ):
        super().__init__()
        self.temperature = temperature
        self.lambda_contrastive = lambda_contrastive
        self.lambda_identity = lambda_identity
        self.lambda_curl = lambda_curl
        self.lambda_div = lambda_div
        self.lambda_orthogonal = lambda_orthogonal
        self.lambda_cycle = lambda_cycle

    def per_attribute_contrastive_loss(
        self,
        z_flowed: torch.Tensor,
        target_attrs: torch.Tensor,
        attr_idx: int
    ) -> torch.Tensor:
        """
        Contrastive loss for a SINGLE attribute.

        Positive pairs: Samples with same value for this attribute
        Negative pairs: Samples with different values

        This avoids the discrete 2^N clustering problem!
        """
        batch_size = z_flowed.size(0)

        # Normalize embeddings
        z_norm = F.normalize(z_flowed, dim=1)

        # Compute similarity matrix
        sim_matrix = torch.matmul(z_norm, z_norm.T) / self.temperature

        # Mask for this specific attribute
        attr_values = target_attrs[:, attr_idx]  # [batch]
        positive_mask = (attr_values.unsqueeze(0) == attr_values.unsqueeze(1)).float()

        # Remove self-similarity
        positive_mask = positive_mask.fill_diagonal_(0)

        # Compute InfoNCE loss for this attribute
        exp_sim = torch.exp(sim_matrix)

        # For each sample, positive = same attribute value
        pos_sim = (exp_sim * positive_mask).sum(dim=1)
        all_sim = exp_sim.sum(dim=1)

        # Avoid log(0)
        loss = -torch.log(pos_sim / (all_sim + 1e-8) + 1e-8)

        return loss.mean()

    def orthogonality_loss(
        self,
        model,
        z: torch.Tensor,
        target_attrs: torch.Tensor
    ) -> torch.Tensor:
        """
        Encourage different attribute vector fields to be orthogonal.

        This prevents interference between attributes.
        """
        num_attrs = target_attrs.size(1)

        # Compute velocity for each attribute separately
        velocities = []
        for attr_idx in range(num_attrs):
            direction = torch.ones(z.size(0), 1, device=z.device)
            v = model.forward_single_attribute(z, attr_idx, direction)
            velocities.append(F.normalize(v, dim=1))

        # Compute pairwise cosine similarities
        loss = 0.0
        count = 0
        for i in range(num_attrs):
            for j in range(i+1, num_attrs):
                # Want velocity_i and velocity_j to be orthogonal
                # cosine_sim should be close to 0
                cos_sim = (velocities[i] * velocities[j]).sum(dim=1).abs()
                loss = loss + cos_sim.mean()
                count += 1

        return loss / count if count > 0 else torch.tensor(0.0, device=z.device)

    def cycle_consistency_loss(
        self,
        model,
        z: torch.Tensor,
        attr_idx: int,
        num_steps: int = 5,
        step_size: float = 0.1
    ) -> torch.Tensor:
        """
        Flow forward then backward should return to start.

        z --[add attribute]--> z' --[remove attribute]--> z''

        z and z'' should be close.
        """
        # Flow forward (add attribute)
        z_forward = model.flow_single_attribute(
            z, attr_idx, target_value=1.0,
            num_steps=num_steps, step_size=step_size
        )

        # Flow backward (remove attribute)
        z_backward = model.flow_single_attribute(
            z_forward, attr_idx, target_value=0.0,
            num_steps=num_steps, step_size=step_size
        )

        # Measure distance (on sphere, use cosine distance)
        z_norm = F.normalize(z, dim=1)
        z_back_norm = F.normalize(z_backward, dim=1)

        cosine_sim = (z_norm * z_back_norm).sum(dim=1)
        loss = 1 - cosine_sim  # cosine distance

        return loss.mean()

    def forward(
        self,
        model,
        z_start: torch.Tensor,
        z_flowed: torch.Tensor,
        original_attrs: torch.Tensor,
        target_attrs: torch.Tensor
    ) -> dict:
        """
        Compute total loss.

        Args:
            model: The vector field model
            z_start: [batch, dim] starting embeddings
            z_flowed: [batch, dim] embeddings after flow
            original_attrs: [batch, num_attrs] original attributes
            target_attrs: [batch, num_attrs] target attributes

        Returns:
            dict with loss components
        """
        losses = {}

        # 1. Per-attribute contrastive loss
        contrastive_loss = 0.0
        for attr_idx in range(target_attrs.size(1)):
            loss = self.per_attribute_contrastive_loss(
                z_flowed, target_attrs, attr_idx
            )
            contrastive_loss = contrastive_loss + loss
        contrastive_loss = contrastive_loss / target_attrs.size(1)
        losses['contrastive'] = contrastive_loss

        # 2. Identity loss (stay near manifold)
        identity_loss = 1 - F.cosine_similarity(z_start, z_flowed, dim=1).mean()
        losses['identity'] = identity_loss

        # 3. Orthogonality loss (attribute independence)
        if self.lambda_orthogonal > 0:
            orthogonal_loss = self.orthogonality_loss(model, z_start, target_attrs)
            losses['orthogonal'] = orthogonal_loss
        else:
            losses['orthogonal'] = torch.tensor(0.0, device=z_start.device)

        # 4. Cycle consistency (optional, expensive)
        if self.lambda_cycle > 0:
            # Only check for first attribute to save compute
            cycle_loss = self.cycle_consistency_loss(model, z_start, attr_idx=0)
            losses['cycle'] = cycle_loss
        else:
            losses['cycle'] = torch.tensor(0.0, device=z_start.device)

        # 5. Curl and divergence (optional)
        # TODO: Add if needed
        losses['curl'] = torch.tensor(0.0, device=z_start.device)
        losses['div'] = torch.tensor(0.0, device=z_start.device)

        # Total loss
        total_loss = (
            self.lambda_contrastive * losses['contrastive'] +
            self.lambda_identity * losses['identity'] +
            self.lambda_orthogonal * losses['orthogonal'] +
            self.lambda_cycle * losses['cycle'] +
            self.lambda_curl * losses['curl'] +
            self.lambda_div * losses['div']
        )
        losses['total'] = total_loss

        return losses
