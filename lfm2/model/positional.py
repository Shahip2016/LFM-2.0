"""Rotary Positional Embeddings (RoPE).

Reference: Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2024).
Applied to Q and K tensors in the GQA blocks.
"""

import torch
import torch.nn as nn


def precompute_freqs_cis(
    dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Precompute the complex exponential frequencies for RoPE.

    Args:
        dim: Head dimension (must be even).
        max_seq_len: Maximum sequence length to precompute for.
        theta: Base frequency for the positional encoding.
        device: Target device.

    Returns:
        Complex tensor of shape ``(max_seq_len, dim // 2)`` containing
        the precomputed frequencies.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # cos + i*sin
    return freqs_cis


def reshape_for_broadcast(
    freqs_cis: torch.Tensor, x: torch.Tensor
) -> torch.Tensor:
    """Reshape frequency tensor for broadcasting with input tensor.

    Args:
        freqs_cis: Precomputed frequencies of shape ``(seq_len, dim//2)``.
        x: Input tensor of shape ``(batch, n_heads, seq_len, dim//2)``.

    Returns:
        Reshaped frequency tensor broadcastable with ``x``.
    """
    ndim = x.ndim
    assert ndim >= 2
    shape = [1] * ndim
    shape[-2] = freqs_cis.shape[0]  # seq_len
    shape[-1] = freqs_cis.shape[1]  # dim // 2
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to query and key tensors.

    Args:
        xq: Query tensor of shape ``(batch, n_heads, seq_len, head_dim)``.
        xk: Key tensor of shape ``(batch, n_kv_heads, seq_len, head_dim)``.
        freqs_cis: Precomputed complex frequencies of shape ``(seq_len, head_dim//2)``.

    Returns:
        Tuple of (rotated queries, rotated keys) with the same shapes as inputs.
    """
    # View as complex numbers: pair consecutive dims
    xq_complex = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_complex = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    freqs_cis_q = reshape_for_broadcast(freqs_cis, xq_complex)
    freqs_cis_k = reshape_for_broadcast(freqs_cis, xk_complex)

    xq_out = torch.view_as_real(xq_complex * freqs_cis_q).flatten(-2)
    xk_out = torch.view_as_real(xk_complex * freqs_cis_k).flatten(-2)

    return xq_out.type_as(xq), xk_out.type_as(xk)


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Positional Embedding module.

    Precomputes and caches the complex exponential frequencies for RoPE,
    and provides a method to apply them to Q/K tensors.

    Args:
        dim: Head dimension (must be even).
        max_seq_len: Maximum sequence length to support.
        theta: Base frequency for the positional encoding.
    """

    def __init__(self, dim: int, max_seq_len: int = 32768, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        freqs_cis = precompute_freqs_cis(dim, max_seq_len, theta)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(
        self, xq: torch.Tensor, xk: torch.Tensor, start_pos: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings to queries and keys.

        Args:
            xq: Query tensor ``(batch, n_heads, seq_len, head_dim)``.
            xk: Key tensor ``(batch, n_kv_heads, seq_len, head_dim)``.
            start_pos: Starting position for the frequency slice.

        Returns:
            Tuple of rotated (queries, keys).
        """
        seq_len = xq.shape[-2]
        freqs = self.freqs_cis[start_pos : start_pos + seq_len]
        return apply_rotary_emb(xq, xk, freqs)
