import unittest
import torch
import torch.nn as nn
from lfm2.model.colbert import LFM2ColBERT, maxsim_score

class DummyConfig:
    d_model = 1024

class DummyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = DummyConfig()
    
    def forward(self, input_ids):
        class Output:
            pass
        out = Output()
        B, L = input_ids.shape
        out.logits = torch.randn(B, L, self.config.d_model)
        return out

class TestColBERT(unittest.TestCase):
    def test_forward_shape(self):
        backbone = DummyBackbone()
        model = LFM2ColBERT(backbone, dim=128, dropout=0.1)
        input_ids = torch.randint(0, 1000, (2, 10))
        embeddings = model(input_ids)
        self.assertEqual(embeddings.shape, (2, 10, 128))

    def test_maxsim_score_shape(self):
        q_emb = torch.randn(2, 5, 128)
        d_emb = torch.randn(2, 20, 128)
        scores = maxsim_score(q_emb, d_emb)
        self.assertEqual(scores.shape, (2, 2))

if __name__ == "__main__":
    unittest.main()
