"""Gated Short Convolution Block.

The core local-context operator in LFM2. Given input h ∈ R^{L×d}:

    Linear: R^d → R^{3d}  →  split into (B, C, h̃)
    y = Conv_k(h̃)           # depthwise 1-D convolution, kernel size k
    z = B ⊙ y               # input-dependent multiplicative gating
    o = z ⊙ C               # second multiplicative gate
    output = Linear_out(o)   # output projection

Wrapped with pre-norm RMSNorm and residual connection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .normalization import RMSNorm


class GatedShortConvolution(nn.Module):
    """Gated Short Convolution operator.

    Provides fast local mixing with excellent cache behavior on CPUs.
    This operator is closely related to the short-range components inside
    many recent efficient sequence blocks (Hyena, Mamba, etc.), but the
    LFM2 architecture search shows that this alone (plus a few GQA blocks)
    is sufficient for the best quality-latency-memory trade-off.

    Args:
        dim: Model hidden dimension.
        kernel_size: Convolution kernel size for local mixing.
        bias: Whether to use bias in linear projections.
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 4,
        bias: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size

        # Input projection: d → 3d (splits into B, C, h_tilde)
        self.input_proj = nn.Linear(dim, 3 * dim, bias=bias)

        # Depthwise 1-D convolution along the sequence dimension
        self.conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            padding=kernel_size - 1,  # causal padding
            groups=dim,  # depthwise
            bias=bias,
        )

        # Output projection: d → d
        self.output_proj = nn.Linear(dim, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(batch, seq_len, dim)``.

        Returns:
            Output tensor of shape ``(batch, seq_len, dim)``.
        """
        batch, seq_len, _ = x.shape

        # Project to 3d and split
        projected = self.input_proj(x)  # (B, L, 3d)
        B_gate, C_gate, h_tilde = projected.chunk(3, dim=-1)  # each (B, L, d)

        # Depthwise causal convolution: transpose to (B, d, L) for Conv1d
        h_conv = h_tilde.transpose(1, 2)  # (B, d, L)
        y = self.conv(h_conv)[:, :, :seq_len]  # causal: trim future
        y = y.transpose(1, 2)  # (B, L, d)

        # Double multiplicative gating
        z = B_gate * y   # input-dependent gating
        o = z * C_gate   # second gate

        # Output projection
        return self.output_proj(o)


class GatedShortConvBlock(nn.Module):
    """LFM2 Gated Short Convolution block with pre-norm and residual.

    Wraps the core ``GatedShortConvolution`` operator with RMSNorm
    pre-normalization and a residual connection, as used in the LFM2 backbone.

    Args:
        dim: Model hidden dimension.
        kernel_size: Convolution kernel size.
        norm_eps: Epsilon for RMSNorm.
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 4,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.norm = RMSNorm(dim, eps=norm_eps)
        self.conv = GatedShortConvolution(dim, kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with pre-norm and residual.

        Args:
            x: Input tensor ``(batch, seq_len, dim)``.

        Returns:
            Output tensor ``(batch, seq_len, dim)``.
        """
        return x + self.conv(self.norm(x))
