"""
Combined loss function for FCLF training.

L_total = L_FCLF + λ_curl * R_curl + λ_div * R_div + λ_identity * ||z_flowed - z_original||^2

The identity loss prevents mode collapse by keeping flowed embeddings close to their originals.
"""

import torch
import torch.nn as nn
from .contrastive_flow_loss import ContrastiveFlowLoss, SimpleContrastiveLoss
from .regularization import CurlRegularization, DivergenceRegularization


class FCLFLoss(nn.Module):
    """
    Complete FCLF loss with contrastive term and regularization.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        alpha: float = 0.1,
        lambda_curl: float = 0.01,
        lambda_div: float = 0.01,
        lambda_identity: float = 0.01,
        curl_epsilon: float = 1e-4,
        curl_samples: int = 10,
        div_epsilon: float = 1e-4
    ):
        """
        Args:
            temperature: Temperature for contrastive loss
            alpha: Flow step size
            lambda_curl: Weight for curl regularization
            lambda_div: Weight for divergence regularization
            lambda_identity: Weight for identity preservation loss
            curl_epsilon: Epsilon for curl finite differences
            curl_samples: Number of random planes for curl estimation
            div_epsilon: Epsilon for divergence finite differences
        """
        super().__init__()

        self.alpha = alpha
        self.lambda_curl = lambda_curl
        self.lambda_div = lambda_div
        self.lambda_identity = lambda_identity

        # Loss components
        self.contrastive_loss = SimpleContrastiveLoss(
            temperature=temperature,
            alpha=alpha
        )

        self.curl_reg = CurlRegularization(
            epsilon=curl_epsilon,
            num_samples=curl_samples
        )

        self.div_reg = DivergenceRegularization(
            epsilon=div_epsilon
        )

    def forward(
        self,
        vector_field,
        z: torch.Tensor,
        y: torch.Tensor,
        return_components: bool = False
    ):
        """
        Compute total FCLF loss.

        Args:
            vector_field: VectorFieldNetwork instance
            z: [batch, dim] CLIP embeddings
            y: [batch, num_attrs] target attributes
            return_components: Whether to return loss components separately

        Returns:
            If return_components=False:
                total_loss: scalar
            If return_components=True:
                (total_loss, contrastive_loss, curl_loss, div_loss)
        """
        # Compute one-step flow
        v = vector_field(z, y)
        z_flowed = z + self.alpha * v

        # Contrastive loss
        contrastive_loss = self.contrastive_loss(z, z_flowed, y)

        # Regularization terms
        curl_loss = self.curl_reg(vector_field, z, y)
        div_loss = self.div_reg(vector_field, z, y)

        # Identity preservation loss (prevents mode collapse)
        # Penalize large deviations from original embedding
        identity_loss = torch.mean((z_flowed - z) ** 2)

        # Total loss
        total_loss = (
            contrastive_loss +
            self.lambda_curl * curl_loss +
            self.lambda_div * div_loss +
            self.lambda_identity * identity_loss
        )

        if return_components:
            return total_loss, contrastive_loss, curl_loss, div_loss, identity_loss
        else:
            return total_loss


class FCLFLossSimple(nn.Module):
    """
    Simplified FCLF loss without expensive regularization terms.
    Useful for faster training/debugging.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        alpha: float = 0.1
    ):
        """
        Args:
            temperature: Temperature for contrastive loss
            alpha: Flow step size
        """
        super().__init__()

        self.alpha = alpha
        self.contrastive_loss = SimpleContrastiveLoss(
            temperature=temperature,
            alpha=alpha
        )

    def forward(
        self,
        vector_field,
        z: torch.Tensor,
        y: torch.Tensor
    ):
        """
        Compute contrastive loss only (no regularization).

        Args:
            vector_field: VectorFieldNetwork instance
            z: [batch, dim] embeddings
            y: [batch, num_attrs] attributes

        Returns:
            loss: scalar
        """
        # One-step flow
        v = vector_field(z, y)
        z_flowed = z + self.alpha * v

        # Contrastive loss
        loss = self.contrastive_loss(z, z_flowed, y)

        return loss
