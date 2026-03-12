"""LFM2-ColBERT: Late-Interaction Retrieval.

Reference: LFM2 Technical Report, Section 7.
Builds on LFM2-350M with task-specific layers and 128-dim projection.
Uses MaxSim operator for fine-grained semantic matching.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class LFM2ColBERT(nn.Module):
    """LFM2-ColBERT Retrieval Model.
    
    A late-interaction model that computes contextualized token embeddings 
    and projects them to a 128-dim space for efficient retrieval.
    """
    
    def __init__(self, backbone: nn.Module, dim: int = 128, dropout: float = 0.0):
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        # Linear layer projects from hidden_dim (1024 for 350M) to 128 dim
        self.proj = nn.Linear(backbone.config.d_model, dim, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encodes query or document into multi-vector representation.
        
        Args:
            input_ids: Token IDs (B, L)
            
        Returns:
            Normalized token embeddings (B, L, 128)
        """
        # 1. Get contextualized token embeddings from backbone
        # We stop before the LM head
        outputs = self.backbone(input_ids)
        h = outputs.logits # Simplification: assuming backbone returns hidden states or logits
        
        # apply dropout
        h = self.dropout(h)
        
        # 2. Project to 128 dimensions
        embeddings = self.proj(h)
        
        # 3. L2 Normalize for cosine similarity
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        
        return embeddings

def maxsim_score(query_embeddings: torch.Tensor, doc_embeddings: torch.Tensor) -> torch.Tensor:
    """Computes the MaxSim score between query and document.
    
    Score = sum_{i in query} max_{j in document} (q_i dot d_j)
    
    Args:
        query_embeddings: (B_q, L_q, D)
        doc_embeddings: (B_d, L_d, D)
        
    Returns:
        Similarity scores (B_q, B_d)
    )"""
    # Compute all-pairs cosine similarity (batch dot product since normalized)
    # (B_q, L_q, D) x (B_d, D, L_d) -> (B_q, B_d, L_q, L_d)
    # Assuming B_q = B_d = B for simplicity here
    B, Lq, D = query_embeddings.shape
    _, Ld, _ = doc_embeddings.shape
    
    # Dot product
    sim_matrix = torch.matmul(query_embeddings, doc_embeddings.transpose(1, 2)) # (B, Lq, Ld)
    
    # MaxSim: max along document tokens, sum along query tokens
    max_sims = sim_matrix.max(dim=-1).values # (B, Lq)
    scores = max_sims.sum(dim=-1) # (B,)
    
    return scores
