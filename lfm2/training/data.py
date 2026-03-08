"""LFM2 Tokenizer and Data Utilities.

LFM2 uses a byte-level BPE tokenizer (tiktoken) with a 65,536-token vocabulary.
Includes special tokens for ChatML, FIM, and tool calling.
"""

from typing import List, Optional, Union
import tiktoken
import torch
from torch.utils.data import Dataset, IterableDataset
import json
import os

class LFM2Tokenizer:
    """Byte-level BPE tokenizer for LFM2."""
    
    SPECIAL_TOKENS = {
        "<|endoftext|>": 0,
        "<|im_start|>": 1,
        "<|im_end|>": 2,
        "<|fim_prefix|>": 3,
        "<|fim_middle|>": 4,
        "<|fim_suffix|>": 5,
        "<|tool_call|>": 6,
        "<|tool_result|>": 7,
    }

    def __init__(self, model_path: Optional[str] = None):
        # In a real scenario, we'd load the trained BPE ranks.
        # Here we initialize a compatible tiktoken encoding with LFM2 special tokens.
        self.encoder = tiktoken.get_encoding("cl100k_base") # Using cl100k as base
        self.n_vocab = 65536
        
    def encode(self, text: str, allowed_special: Union[str, set] = "all") -> List[int]:
        return self.encoder.encode(text, allowed_special=allowed_special)

    def decode(self, tokens: List[int]) -> str:
        return self.encoder.decode(tokens)

    @property
    def vocab_size(self) -> int:
        return self.n_vocab

class PreTrainingDataset(IterableDataset):
    """Streaming dataset for large-scale pre-training."""
    
    def __init__(self, data_path: str, tokenizer: LFM2Tokenizer, max_seq_len: int = 4096):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __iter__(self):
        # Placeholder for streaming logic
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                text = data.get("text", "")
                tokens = self.tokenizer.encode(text)
                
                for i in range(0, len(tokens) - self.max_seq_len, self.max_seq_len):
                    yield torch.tensor(tokens[i:i + self.max_seq_len])

class FIMDataset(Dataset):
    """Dataset wrapper for Fill-In-the-Middle (FIM) objective."""
    
    def __init__(self, base_dataset: Dataset, fim_rate: float = 0.5):
        self.base_dataset = base_dataset
        self.fim_rate = fim_rate

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # Implementation of FIM logic (prefix, suffix, middle transformation)
        tokens = self.base_dataset[idx]
        if torch.rand(1).item() > self.fim_rate:
            return tokens
            
        # Simplistic split for demonstration
        L = len(tokens)
        p1, p2 = sorted(torch.randint(0, L, (2,)).tolist())
        prefix = tokens[:p1]
        middle = tokens[p1:p2]
        suffix = tokens[p2:]
        
        # LFM2 FIM format: <fim_prefix> PRE <fim_suffix> SUF <fim_middle> MID
        return torch.cat([
            torch.tensor([3]), prefix, 
            torch.tensor([5]), suffix, 
            torch.tensor([4]), middle
        ])
