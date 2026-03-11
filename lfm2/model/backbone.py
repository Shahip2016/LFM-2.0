"""LFM2 Hybrid Backbone — Model Assembly.

Assembles the full LFM2 model from gated short convolution blocks and
GQA blocks, based on a configurable block pattern. Includes token
embedding, final normalization, and LM head.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import GQABlock, KVCache
from .conv_block import GatedShortConvBlock
from .normalization import RMSNorm
from typing import Optional


@dataclass
class LFM2Config:
    """Configuration for an LFM2 dense model.

    Args:
        n_layers: Total number of blocks in the backbone.
        d_model: Hidden dimension of the model.
        n_heads: Number of query heads in GQA blocks.
        n_kv_heads: Number of key-value heads in GQA blocks.
        head_dim: Dimension per attention head.
        mlp_hidden_dim: Hidden dimension for SwiGLU MLP (None → auto).
        conv_kernel_size: Kernel size for gated short convolution.
        vocab_size: Vocabulary size for the BPE tokenizer.
        max_seq_len: Maximum sequence length (context window).
        block_pattern: List of ``"conv"`` or ``"attn"`` strings defining
            the block type at each layer. Length must equal ``n_layers``.
        norm_eps: Epsilon for RMSNorm.
        dropout: Dropout probability for attention.
        tie_word_embeddings: Whether to tie input and output embeddings.
    """

    n_layers: int = 16
    d_model: int = 1024
    n_heads: int = 16
    n_kv_heads: int = 4
    head_dim: int = 64
    mlp_hidden_dim: int | None = None
    conv_kernel_size: int = 4
    vocab_size: int = 65536
    max_seq_len: int = 32768
    block_pattern: list[str] | None = None
    norm_eps: float = 1e-6
    dropout: float = 0.0
    tie_word_embeddings: bool = True


class LFM2Model(nn.Module):
    """LFM2 Language Model.

    A decoder-only model with the LFM2 hybrid backbone architecture:
    majority gated short convolution blocks interleaved with a minority
    of GQA blocks.

    Args:
        config: Model configuration.
    """

    def __init__(self, config: LFM2Config):
        super().__init__()
        self.config = config

        # Token embedding
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)

        # Build block pattern
        if config.block_pattern is None:
            # Default: 10 conv + 6 attn for 16 layers (as per LFM2-350M)
            pattern = self._default_pattern(config.n_layers)
        else:
            pattern = config.block_pattern
            assert len(pattern) == config.n_layers

        # Build layers
        self.layers = nn.ModuleList()
        for block_type in pattern:
            if block_type == "conv":
                self.layers.append(
                    GatedShortConvBlock(
                        dim=config.d_model,
                        kernel_size=config.conv_kernel_size,
                        norm_eps=config.norm_eps,
                    )
                )
            elif block_type == "attn":
                self.layers.append(
                    GQABlock(
                        dim=config.d_model,
                        n_heads=config.n_heads,
                        n_kv_heads=config.n_kv_heads,
                        head_dim=config.head_dim,
                        mlp_hidden_dim=config.mlp_hidden_dim,
                        max_seq_len=config.max_seq_len,
                        norm_eps=config.norm_eps,
                        dropout=config.dropout,
                    )
                )
            else:
                raise ValueError(f"Unknown block type: {block_type}")

        # Final norm and LM head
        self.final_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Tie weights
        if config.tie_word_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        # Initialize weights
        self.apply(self._init_weights)

    @staticmethod
    def _default_pattern(n_layers: int) -> list[str]:
        """Generate default block pattern: mostly conv, ~37.5% attn evenly spaced."""
        n_attn = max(1, n_layers * 6 // 16)
        pattern = ["conv"] * n_layers
        # Space attention layers evenly
        if n_attn > 0:
            step = n_layers / n_attn
            for i in range(n_attn):
                idx = int(round((i + 0.5) * step))
                idx = min(idx, n_layers - 1)
                pattern[idx] = "attn"
        return pattern

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights with small normal distribution."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _create_causal_mask(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Create a causal attention mask."""
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype)
        mask = torch.triu(mask, diagonal=1)
        return mask

    def forward(
        self,
        input_ids: torch.Tensor,
        start_pos: int = 0,
        kv_caches: list[Optional[KVCache]] | None = None,
        labels: torch.Tensor | None = None,
    ) -> "LFM2Output":
        """Forward pass.

        Args:
            input_ids: Token IDs of shape ``(batch, seq_len)``.
            start_pos: Starting position for RoPE.
            kv_caches: Optional list of KVCache objects, one for each attention layer.
            labels: Optional labels for next-token prediction loss.

        Returns:
            ``LFM2Output`` with logits, optional loss, and updated caches.
        """
        batch, seq_len = input_ids.shape
        device = input_ids.device

        # Embed tokens
        h = self.tok_emb(input_ids)

        # Create causal mask for attention layers
        mask = None
        if seq_len > 1:
            mask = self._create_causal_mask(seq_len, device, h.dtype)

        # Pass through all layers
        new_kv_caches = []
        attn_layer_idx = 0
        for layer in self.layers:
            if isinstance(layer, GQABlock):
                current_cache = kv_caches[attn_layer_idx] if kv_caches is not None else None
                h, cache = layer(h, start_pos=start_pos, mask=mask, kv_cache=current_cache)
                new_kv_caches.append(cache)
                attn_layer_idx += 1
            else:
                h = layer(h)

        # Final norm and LM head
        h = self.final_norm(h)
        logits = self.lm_head(h)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return LFM2Output(logits=logits, loss=loss, kv_caches=new_kv_caches if new_kv_caches else None)

    @classmethod
    def from_config(cls, config: LFM2Config) -> "LFM2Model":
        """Factory method to create a model from a config."""
        return cls(config)

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


@dataclass
class LFM2Output:
    """Output from the LFM2 model.

    Args:
        logits: Token logits of shape ``(batch, seq_len, vocab_size)``.
        loss: Optional next-token prediction loss.
    """

    logits: torch.Tensor
    loss: torch.Tensor | None = None
    kv_caches: list[Optional[KVCache]] | None = None
