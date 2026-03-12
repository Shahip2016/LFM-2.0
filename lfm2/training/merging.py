"""LFM2 Model Merging Utilities.

Reference: LFM2 Technical Report, Appendix B.
Implements Model Soup, Task Arithmetic, TIES-Merging, DARE, and DELLA.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional

def model_soup(models_states: List[Dict[str, torch.Tensor]], weights: List[float]) -> Dict[str, torch.Tensor]:
    """Weighted averaging of model parameters."""
    merged = {}
    for key in models_states[0].keys():
        merged[key] = sum(m[key] * w for m, w in zip(models_states, weights))
    return merged

def simple_average(models_states: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Averages parameters equally across models without specified weights."""
    merged = {}
    num_models = len(models_states)
    for key in models_states[0].keys():
        merged[key] = sum(m[key] for m in models_states) / num_models
    return merged

def task_arithmetic(
    base_state: Dict[str, torch.Tensor],
    models_states: List[Dict[str, torch.Tensor]],
    weights: List[float]
) -> Dict[str, torch.Tensor]:
    """Delta-based merging: base + sum(weight_i * (model_i - base))."""
    merged = {k: v.clone() for k, v in base_state.items()}
    for k in merged.keys():
        for m, w in zip(models_states, weights):
            delta = m[k] - base_state[k]
            merged[k] += w * delta
    return merged

def ties_merge(
    base_state: Dict[str, torch.Tensor],
    models_states: List[Dict[str, torch.Tensor]],
    weights: List[float],
    density: float = 0.2
) -> Dict[str, torch.Tensor]:
    """ Trim, Elect Sign & Merge. Resolves parameter interference. """
    merged = {k: v.clone() for k, v in base_state.items()}
    
    for k in merged.keys():
        deltas = []
        for m in models_states:
            delta = m[k] - base_state[k]
            
            # 1. Trim: sparsify delta by magnitude
            if delta.dim() > 0:
                top_k = int(density * delta.numel())
                if top_k > 0:
                    thres = torch.topk(delta.abs().view(-1), top_k).values[-1]
                    delta = torch.where(delta.abs() >= thres, delta, torch.zeros_like(delta))
            deltas.append(delta)
            
        # 2. Elect Sign
        stacked_deltas = torch.stack(deltas)
        signs = torch.sign(stacked_deltas.sum(dim=0))
        
        # 3. Disjoint Merge (agree with majority sign)
        mask = (torch.sign(stacked_deltas) == signs).float()
        weighted_deltas = stacked_deltas * mask * torch.tensor(weights).view(-1, *([1]*delta.dim()))
        
        # Avoid division by zero
        num_agree = mask.sum(dim=0)
        sum_delta = weighted_deltas.sum(dim=0) / (num_agree + 1e-8)
        
        merged[k] += sum_delta
        
    return merged

def dare_merge(
    base_state: Dict[str, torch.Tensor],
    models_states: List[Dict[str, torch.Tensor]],
    weights: List[float],
    drop_rate: float = 0.5
) -> Dict[str, torch.Tensor]:
    """ Drop And REscale. Random sparsification for better merging. """
    merged = {k: v.clone() for k, v in base_state.items()}
    
    for k in merged.keys():
        rescale = 1.0 / (1.0 - drop_rate)
        for m, w in zip(models_states, weights):
            delta = m[k] - base_state[k]
            # Random dropout
            mask = (torch.rand_like(delta) > drop_rate).float()
            merged[k] += w * (delta * mask * rescale)
            
    return merged

def della_merge(
    base_state: Dict[str, torch.Tensor],
    models_states: List[Dict[str, torch.Tensor]],
    weights: List[float],
    drop_rate: float = 0.5,
    epsilon: float = 0.1
) -> Dict[str, torch.Tensor]:
    """ Drop and rEscaLe via sampLing with mAgnitude. """
    # Simplified DELLA: magnitude-aware dropout
    merged = {k: v.clone() for k, v in base_state.items()}
    
    for k in merged.keys():
        for m, w in zip(models_states, weights):
            delta = m[k] - base_state[k]
            if delta.dim() == 0:
                merged[k] += w * delta
                continue
                
            # Assign higher dropout to lower magnitude parameters
            ranks = torch.argsort(torch.argsort(delta.abs().view(-1)))
            ranks = ranks.view_as(delta).float() / delta.numel()
            
            # dropout prob inversely proportional to rank magnitude
            p_dropout = drop_rate + epsilon * (0.5 - ranks)
            p_dropout = torch.clamp(p_dropout, 0.0, 0.99)
            
            mask = (torch.rand_like(delta) > p_dropout).float()
            rescale = 1.0 / (1.0 - p_dropout)
            merged[k] += w * (delta * mask * rescale)
            
    return merged
