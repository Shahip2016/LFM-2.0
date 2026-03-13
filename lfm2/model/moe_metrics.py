import torch
import torch.nn as nn
from typing import Dict, Any

class MoEMetricsTracker:
    """
    Tracks and computes gating metrics for Mixture of Experts (MoE) layers.
    """
    def __init__(self, num_experts: int):
        self.num_experts = num_experts
        self.reset()
        
    def reset(self):
        """Resets the running metrics."""
        self.total_tokens = 0
        self.expert_counts = [0] * self.num_experts
        self.routing_probs_sum = [0.0] * self.num_experts

    def update(self, routing_probs: torch.Tensor, expert_indices: torch.Tensor):
        """
        Updates metrics with routing information from a forward pass.
        
        Args:
            routing_probs: Tensor of shape (batch * seq_len, num_experts).
            expert_indices: Tensor of shape (batch * seq_len, top_k) indicating selected experts.
        """
        batch_seq_len = routing_probs.shape[0]
        self.total_tokens += batch_seq_len
        
        for k in range(expert_indices.shape[1]):
            indices = expert_indices[:, k]
            counts = torch.bincount(indices, minlength=self.num_experts)
            for i in range(self.num_experts):
                self.expert_counts[i] += counts[i].item()
                
        probs_sum = routing_probs.sum(dim=0)
        for i in range(self.num_experts):
            self.routing_probs_sum[i] += probs_sum[i].item()

    def get_metrics(self) -> Dict[str, Any]:
        """Returns the computed metrics."""
        return {
            'expert_counts': self.expert_counts,
            'routing_probs_sum': self.routing_probs_sum,
            'total_tokens': self.total_tokens
        }
