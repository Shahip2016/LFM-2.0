"""Mixture of Experts (MoE) Layer.

LFM2-8B-A1B replaces dense SwiGLU MLPs with sparse MoE MLPs in most layers.
- 32 experts per MoE layer, Top-k=4 selected per token
- First 2 layers remain dense for stability
- Normalized sigmoid router with adaptive load-balancing biases

Reference: Liu et al., "DeepSeek-V2" (2024) for the routing mechanism.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .activations import SwiGLU
from .normalization import RMSNorm


class SigmoidRouter(nn.Module):
    """Normalized sigmoid router for MoE expert selection.

    Uses sigmoid activation followed by normalization, with adaptive
    per-expert biases for load balancing (following DeepSeek-V2).

    Args:
        dim: Input feature dimension.
        n_experts: Total number of experts.
        top_k: Number of experts to select per token.
    """

    def __init__(self, dim: int, n_experts: int, top_k: int = 4):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k

        self.gate = nn.Linear(dim, n_experts, bias=False)
        # Adaptive load-balancing biases
        self.expert_bias = nn.Parameter(torch.zeros(n_experts))

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute routing weights and expert assignments.

        Args:
            x: Input tensor of shape ``(batch * seq_len, dim)``.

        Returns:
            Tuple of:
            - ``weights``: Routing weights for selected experts ``(tokens, top_k)``.
            - ``indices``: Indices of selected experts ``(tokens, top_k)``.
            - ``load_balance_loss``: Auxiliary loss for load balancing.
        """
        # Compute router logits with bias
        logits = self.gate(x) + self.expert_bias  # (tokens, n_experts)

        # Normalized sigmoid routing
        scores = torch.sigmoid(logits)
        scores = scores / (scores.sum(dim=-1, keepdim=True) + 1e-6)

        # Top-k selection
        weights, indices = torch.topk(scores, self.top_k, dim=-1)

        # Normalize selected weights
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-6)

        # Load balancing loss: encourage uniform expert utilization
        load_balance_loss = self._compute_load_balance_loss(scores, indices)

        return weights, indices, load_balance_loss

    def _compute_load_balance_loss(
        self, scores: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        """Compute auxiliary load balance loss.

        Encourages each expert to receive a roughly equal number of tokens.

        Args:
            scores: Full routing scores ``(tokens, n_experts)``.
            indices: Selected expert indices ``(tokens, top_k)``.

        Returns:
            Scalar load balancing loss.
        """
        n_tokens = scores.shape[0]

        # Fraction of tokens routed to each expert
        one_hot = F.one_hot(indices, self.n_experts).float()  # (tokens, top_k, n_experts)
        tokens_per_expert = one_hot.sum(dim=(0, 1)) / n_tokens  # (n_experts,)

        # Average routing probability per expert
        avg_prob = scores.mean(dim=0)  # (n_experts,)

        # Loss = n_experts * sum(tokens_per_expert * avg_prob)
        loss = self.n_experts * (tokens_per_expert * avg_prob).sum()
        return loss


class MoELayer(nn.Module):
    """Mixture of Experts layer.

    Routes each token to its Top-k experts and combines their outputs
    using the routing weights from the sigmoid router.

    Args:
        dim: Hidden dimension.
        n_experts: Number of experts.
        top_k: Number of experts per token.
        expert_hidden_dim: Hidden dimension for each expert's SwiGLU MLP.
    """

    def __init__(
        self,
        dim: int,
        n_experts: int = 32,
        top_k: int = 4,
        expert_hidden_dim: int | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.n_experts = n_experts
        self.top_k = top_k

        # Router
        self.router = SigmoidRouter(dim, n_experts, top_k)

        # Expert MLPs (each is a SwiGLU)
        self.experts = nn.ModuleList([
            SwiGLU(dim, hidden_dim=expert_hidden_dim) for _ in range(n_experts)
        ])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with expert routing.

        Args:
            x: Input tensor of shape ``(batch, seq_len, dim)``.

        Returns:
            Tuple of:
            - Output tensor ``(batch, seq_len, dim)``.
            - Load balance loss (scalar).
        """
        batch, seq_len, dim = x.shape
        x_flat = x.view(-1, dim)  # (B*L, d)

        # Route
        weights, indices, lb_loss = self.router(x_flat)  # weights/indices: (B*L, top_k)

        # Compute expert outputs
        output = torch.zeros_like(x_flat)

        for i, expert in enumerate(self.experts):
            # Find tokens routed to this expert
            mask = (indices == i).any(dim=-1)  # (B*L,)
            if not mask.any():
                continue

            # Get the weight for this expert for each token
            expert_weight = (weights * (indices == i).float()).sum(dim=-1)  # (B*L,)

            # Compute expert output for selected tokens
            expert_input = x_flat[mask]
            expert_output = expert(expert_input)

            # Weighted accumulation
            output[mask] += expert_weight[mask].unsqueeze(-1) * expert_output

        return output.view(batch, seq_len, dim), lb_loss


class MoEBlock(nn.Module):
    """LFM2 MoE block replacing the dense MLP with sparse MoE.

    Used in the LFM2-8B-A1B model. For GQA blocks, replaces the SwiGLU
    MLP with an MoE layer. For conv blocks, adds MoE as an additional sublayer.

    Args:
        dim: Hidden dimension.
        n_experts: Number of experts.
        top_k: Number of active experts per token.
        expert_hidden_dim: Hidden dimension per expert.
        norm_eps: Epsilon for RMSNorm.
    """

    def __init__(
        self,
        dim: int,
        n_experts: int = 32,
        top_k: int = 4,
        expert_hidden_dim: int | None = None,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.norm = RMSNorm(dim, eps=norm_eps)
        self.moe = MoELayer(
            dim=dim,
            n_experts=n_experts,
            top_k=top_k,
            expert_hidden_dim=expert_hidden_dim,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor ``(batch, seq_len, dim)``.

        Returns:
            Tuple of (output tensor, load balance loss).
        """
        residual = x
        h = self.norm(x)
        moe_out, lb_loss = self.moe(h)
        return residual + moe_out, lb_loss
