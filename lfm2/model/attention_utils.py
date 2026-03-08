"""Attention utility modules.

QK-Norm: L2-normalization of Q and K before the dot product.
Reference: Dehghani et al., "Scaling Vision Transformers to 22B Parameters" (2023).
"""

import torch
import torch.nn as nn


class QKNorm(nn.Module):
    """QK-Norm: L2-normalize queries and keys before attention.

    Applying L2 normalization to Q and K stabilizes training at scale
    by preventing the attention logits from growing unboundedly.

    Args:
        head_dim: Dimension of each attention head.
    """

    def __init__(self, head_dim: int):
        super().__init__()
        self.head_dim = head_dim
        self.q_norm = nn.LayerNorm(head_dim, elementwise_affine=False)
        self.k_norm = nn.LayerNorm(head_dim, elementwise_affine=False)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply L2-normalization to query and key tensors.

        Args:
            q: Query tensor of shape ``(..., head_dim)``.
            k: Key tensor of shape ``(..., head_dim)``.

        Returns:
            Tuple of (normalized_q, normalized_k).
        """
        return self.q_norm(q), self.k_norm(k)
