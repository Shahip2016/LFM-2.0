"""Grouped Query Attention (GQA).

Reference: Ainslie et al., "GQA: Training Generalized Multi-Query Transformer
Models from Multi-Head Checkpoints" (2023).

GQA reduces KV traffic by sharing keys/values across head groups while
preserving multi-head queries. Augmented with RoPE and QK-Norm as per the
LFM2 architecture.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention_utils import QKNorm
from .normalization import RMSNorm
from .positional import RotaryPositionalEmbedding


class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention with RoPE and QK-Norm.

    Args:
        dim: Model hidden dimension.
        n_heads: Number of query heads.
        n_kv_heads: Number of key-value heads (must divide ``n_heads``).
        head_dim: Dimension per head.
        max_seq_len: Maximum sequence length for RoPE precomputation.
        dropout: Attention dropout probability.
        bias: Whether to use bias in projections.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        max_seq_len: int = 32768,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.n_rep = n_heads // n_kv_heads  # repetition factor for KV heads
        self.scale = head_dim ** -0.5

        # Projections
        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=bias)

        # QK-Norm and RoPE
        self.qk_norm = QKNorm(head_dim)
        self.rope = RotaryPositionalEmbedding(head_dim, max_seq_len=max_seq_len)

        self.attn_dropout = nn.Dropout(dropout)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Repeat KV heads to match the number of query heads.

        Args:
            x: Tensor of shape ``(batch, n_kv_heads, seq_len, head_dim)``.

        Returns:
            Tensor of shape ``(batch, n_heads, seq_len, head_dim)``.
        """
        if self.n_rep == 1:
            return x
        batch, n_kv_heads, seq_len, head_dim = x.shape
        x = x[:, :, None, :, :].expand(batch, n_kv_heads, self.n_rep, seq_len, head_dim)
        return x.reshape(batch, self.n_heads, seq_len, head_dim)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int = 0,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(batch, seq_len, dim)``.
            start_pos: Starting position for RoPE (used in KV-cache inference).
            mask: Optional attention mask of shape ``(seq_len, seq_len)``
                or ``(batch, 1, seq_len, seq_len)``.

        Returns:
            Output tensor of shape ``(batch, seq_len, dim)``.
        """
        batch, seq_len, _ = x.shape

        # Project Q, K, V
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)

        # Transpose to (batch, heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply QK-Norm
        q, k = self.qk_norm(q, k)

        # Apply RoPE
        q, k = self.rope(q, k, start_pos=start_pos)

        # Repeat KV heads if GQA
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn_weights = attn_weights + mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(q)
        attn_weights = self.attn_dropout(attn_weights)

        # Compute attention output
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.o_proj(attn_output)


class GQABlock(nn.Module):
    """LFM2 GQA block with pre-norm, attention, and MLP.

    Each GQA block consists of:
    1. RMSNorm → GroupedQueryAttention (with RoPE, QK-Norm)
    2. RMSNorm → SwiGLU MLP

    Both sublayers use residual connections.

    Args:
        dim: Model hidden dimension.
        n_heads: Number of query heads.
        n_kv_heads: Number of key-value heads.
        head_dim: Dimension per head.
        mlp_hidden_dim: Hidden dimension for the SwiGLU MLP.
        max_seq_len: Maximum sequence length.
        norm_eps: Epsilon for RMSNorm.
        dropout: Attention dropout probability.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        mlp_hidden_dim: int | None = None,
        max_seq_len: int = 32768,
        norm_eps: float = 1e-6,
        dropout: float = 0.0,
    ):
        super().__init__()
        from .activations import SwiGLU

        # Attention sublayer
        self.attn_norm = RMSNorm(dim, eps=norm_eps)
        self.attn = GroupedQueryAttention(
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )

        # MLP sublayer
        self.mlp_norm = RMSNorm(dim, eps=norm_eps)
        self.mlp = SwiGLU(dim, hidden_dim=mlp_hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int = 0,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward with pre-norm and residual connections.

        Args:
            x: Input tensor ``(batch, seq_len, dim)``.
            start_pos: Starting position for RoPE.
            mask: Optional attention mask.

        Returns:
            Output tensor ``(batch, seq_len, dim)``.
        """
        # Attention sublayer
        x = x + self.attn(self.attn_norm(x), start_pos=start_pos, mask=mask)
        # MLP sublayer
        x = x + self.mlp(self.mlp_norm(x))
        return x
