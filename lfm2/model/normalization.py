"""RMSNorm — Root Mean Square Layer Normalization.

Reference: Zhang & Sennrich, "Root Mean Square Layer Normalization" (2019).
Used as pre-norm in every LFM2 block.
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Normalizes the input by its RMS value, with a learnable scale parameter.
    More efficient than LayerNorm as it omits the mean-centering step.

    Args:
        dim: Feature dimension to normalize over.
        eps: Small constant for numerical stability.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMSNorm.

        Args:
            x: Input tensor of shape ``(..., dim)``.

        Returns:
            Normalized tensor of the same shape.
        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
