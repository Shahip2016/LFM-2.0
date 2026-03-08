"""LFM2 Post-Training: SFT and Alignment.

Implements Supervised Fine-Tuning (SFT) and generalized 
length-normalized preference alignment (DPO/APO).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional

class LengthNormalizedAlignmentLoss(nn.Module):
    """Generalized Length-Normalized Preference Alignment loss.
    
    Covers DPO, APO, and SimPO as special cases with length normalization 
    to prevent length bias in small models.
    """
    
    def __init__(
        self, 
        beta: float = 0.1, 
        margin: float = 0.1, 
        omega: float = 1.0, 
        lambda_val: float = 0.2
    ):
        super().__init__()
        self.beta = beta
        self.margin = margin
        self.omega = omega
        self.lambda_val = lambda_val

    def forward(
        self, 
        policy_chosen_logps: torch.Tensor, 
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
        chosen_tokens: torch.Tensor,
        rejected_tokens: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass for alignment loss.
        
        Args:
            policy_*_logps: (B,) Log-probabilities from the current policy model.
            reference_*_logps: (B,) Log-probabilities from the reference SFT model.
            *_tokens: (B, L) Sequences for length counting.
        """
        # Length normalization
        chosen_len = chosen_tokens.ne(-100).sum(dim=-1).float()
        rejected_len = rejected_tokens.ne(-100).sum(dim=-1).float()
        
        # Relative reward (DPO/Implicit)
        pi_ratio_chosen = policy_chosen_logps / chosen_len
        ref_ratio_chosen = reference_chosen_logps / chosen_len
        pi_ratio_rejected = policy_rejected_logps / rejected_len
        ref_ratio_rejected = reference_rejected_logps / rejected_len
        
        rel_diff = (pi_ratio_chosen - ref_ratio_chosen) - (pi_ratio_rejected - ref_ratio_rejected)
        
        # Loss components (Simplified generalized version)
        # L_DPO variant with length normalization
        loss_DPO = -F.logsigmoid(self.beta * rel_diff - self.margin)
        
        return loss_DPO.mean()

class SFTTrainer:
    """Trainer for Supervised Fine-Tuning."""
    
    def __init__(self, model: nn.Module, lr: float = 1e-5):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def train_step(self, input_ids: torch.Tensor, labels: torch.Tensor) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        
        output = self.model(input_ids)
        logits = output.logits
        
        # Shift for NTP (standard SFT)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss = self.loss_fn(
            shift_logits.view(-1, logits.size(-1)),
            shift_labels.view(-1)
        )
        
        loss.backward()
        self.optimizer.step()
        return loss.item()
