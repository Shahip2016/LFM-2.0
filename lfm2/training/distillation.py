"""Decoupled Top-K Knowledge Distillation (DTK-KD).

Reference: LFM2 Technical Report, Section 3.3.
Decouples the KL divergence into a binary term (mass matching) and 
a conditional term (distribution matching within Top-K).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoupledTopKDistillationLoss(nn.Module):
    """Implementation of Decoupled Top-K Knowledge Distillation.
    
    This objective avoids support mismatch between teacher and student 
    truncated distributions, which is especially critical for small models 
    trained with high-temperature distillation.
    """
    
    def __init__(self, top_k: int = 32, temperature: float = 2.0):
        super().__init__()
        self.top_k = top_k
        self.temperature = temperature

    def forward(self, student_logits, teacher_logits):
        """Forward pass for DTK-KD loss.
        
        Args:
            student_logits: Logits from the student model (B, V) or (B, L, V)
            teacher_logits: Logits from the teacher model (B, V) or (B, L, V)
        """
        if student_logits.dim() == 3:
            student_logits = student_logits.view(-1, student_logits.size(-1))
            teacher_logits = teacher_logits.view(-1, teacher_logits.size(-1))

        # 1. Identify Teacher's Top-K set T
        topk_teacher_probs, topk_indices = torch.topk(
            F.softmax(teacher_logits, dim=-1), 
            self.top_k, 
            dim=-1
        )
        
        # Total mass on Top-K set
        p_t_T = topk_teacher_probs.sum(dim=-1) # (tokens,)
        
        # 2. Binary Term L_B: Match total mass on Top-K set
        # Success probability for Bernoulli: mass on Top-K
        student_probs = F.softmax(student_logits, dim=-1)
        p_s_T = torch.gather(student_probs, 1, topk_indices).sum(dim=-1)
        
        # Binary KL (expressed as BCE for mass targets)
        loss_B = F.binary_cross_entropy(p_s_T, p_t_T)

        # 3. Conditional Term L_T: Temperature-scaled matching within Top-K
        # Conditional student/teacher distributions within T
        # p(x | token is in T) = p(x) / sum_{x' in T} p(x')
        
        # Gather Top-K student/teacher logits
        topk_student_logits = torch.gather(student_logits, 1, topk_indices)
        topk_teacher_logits = torch.gather(teacher_logits, 1, topk_indices)
        
        # Apply temperature only to conditional KL
        loss_T = F.kl_div(
            F.log_softmax(topk_student_logits / self.temperature, dim=-1),
            F.softmax(topk_teacher_logits / self.temperature, dim=-1),
            reduction="batchmean"
        )
        
        # 4. Final DTK-KD loss (untempered mass matching + scalar-weighted conditional)
        # Scaled by temperature^2 to balance signal strength
        return loss_B + p_t_T.mean() * (self.temperature ** 2) * loss_T
