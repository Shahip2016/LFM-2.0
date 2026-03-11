"""Demo script for LFM2-ColBERT late-interaction retrieval.

This script demonstrates how to use the LFM2-ColBERT model for fine-grained
semantic matching between a query and a set of documents.
"""

import torch
import torch.nn as nn
from lfm2.model.backbone import LFM2Model, LFM2Config
from lfm2.model.colbert import LFM2ColBERT, maxsim_score

def run_demo():
    # 1. Initialize Model
    config = LFM2Config(n_layers=4, d_model=256, n_heads=4, n_kv_heads=2)
    backbone = LFM2Model(config)
    colbert_model = LFM2ColBERT(backbone, dim=128)
    colbert_model.eval()

    # 2. Sample Data
    query = "How do liquid foundation models handle long sequences?"
    documents = [
        "Liquid Foundation Models (LFMs) use gated short convolutions to achieve linear complexity in sequence length.",
        "The GQA mechanism in LFMs helps reduce KV traffic while maintaining high performance on attention tasks.",
        "Traditional Transformers have quadratic complexity, making them slower on very long contexts compared to LFMs."
    ]

    # Mock tokenizer (for demo purposes)
    def mock_tokenize(text, max_len=32):
        # In a real scenario, use lfm2.tokenizer
        tokens = torch.randint(0, config.vocab_size, (1, max_len))
        return tokens

    # 3. Encode Query
    query_ids = mock_tokenize(query)
    query_enc = colbert_model(query_ids) # (1, Lq, 128)

    # 4. Encode Documents
    doc_scores = []
    print(f"Query: {query}\n")
    print("Ranking results:")
    
    for i, doc in enumerate(documents):
        doc_ids = mock_tokenize(doc)
        doc_enc = colbert_model(doc_ids) # (1, Ld, 128)
        
        # 5. Compute MaxSim Score
        score = maxsim_score(query_enc, doc_enc).item()
        doc_scores.append((score, doc))

    # Sort by score descending
    doc_scores.sort(key=lambda x: x[0], reverse=True)

    for i, (score, doc) in enumerate(doc_scores):
        print(f"{i+1}. [Score: {score:.4f}] {doc}")

if __name__ == "__main__":
    run_demo()
