import torch
import unittest
from lfm2.model.generation import generate_sequences

class DummyModel(torch.nn.Module):
    def __init__(self, vocab_size=10):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, x):
        batch_size, seq_len = x.shape
        # Return random logits for the vocabulary
        return torch.randn(batch_size, seq_len, self.vocab_size)

class TestGeneration(unittest.TestCase):
    def test_generate_sequences(self):
        model = DummyModel()
        input_ids = torch.tensor([[1, 2, 3]])
        max_length = 5
        generated = generate_sequences(model, input_ids, max_length=max_length)
        
        # The output length should be the input length (3) + max_length (5)
        self.assertEqual(generated.shape, (1, 8))

if __name__ == '__main__':
    unittest.main()
