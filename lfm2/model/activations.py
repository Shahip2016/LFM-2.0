"""Activation functions and feed-forward blocks.

Reference: Shazeer, "GLU Variants Improve Transformer" (2020).
SwiGLU is used as the position-wise MLP in all LFM2 layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """SwiGLU feed-forward block.

    Computes: ``output = W_down · (SiLU(W_gate · x) ⊙ (W_up · x))``

    The expansion ratio controls the intermediate hidden dimension.
    The paper uses size-dependent expansion ratios chosen by architecture search.

    Args:
        dim: Input and output feature dimension.
        hidden_dim: Intermediate hidden dimension. If ``None``, defaults to
            ``int(dim * expansion_ratio)``.
        expansion_ratio: Expansion ratio for computing ``hidden_dim`` when
            ``hidden_dim`` is not provided. Default: ``8/3`` (~2.67x).
        bias: Whether to use bias in linear layers.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int | None = None,
        expansion_ratio: float = 8 / 3,
        bias: bool = False,
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(dim * expansion_ratio)
            # Round to nearest multiple of 64 for efficiency
            hidden_dim = ((hidden_dim + 63) // 64) * 64

        self.w_gate = nn.Linear(dim, hidden_dim, bias=bias)
        self.w_up = nn.Linear(dim, hidden_dim, bias=bias)
        self.w_down = nn.Linear(hidden_dim, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(..., dim)``.

        Returns:
            Output tensor of the same shape.
        """
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))

class GeGLU(nn.Module):
    """GeGLU feed-forward block.

    Computes: ``output = W_down · (GELU(W_gate · x) ⊙ (W_up · x))``
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int | None = None,
        expansion_ratio: float = 8 / 3,
        bias: bool = False,
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(dim * expansion_ratio)
            hidden_dim = ((hidden_dim + 63) // 64) * 64

        self.w_gate = nn.Linear(dim, hidden_dim, bias=bias)
        self.w_up = nn.Linear(dim, hidden_dim, bias=bias)
        self.w_down = nn.Linear(hidden_dim, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.w_down(F.gelu(self.w_gate(x)) * self.w_up(x))
