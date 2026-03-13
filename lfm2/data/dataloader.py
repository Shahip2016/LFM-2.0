import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict
import numpy as np

class PreTrainingDataset(Dataset):
    """
    Dataset for LFM pre-training. Handles tokenized text data.
    """
    def __init__(self, data_path: str, seq_len: int = 2048):
        self.data_path = data_path
        self.seq_len = seq_len
        # In a real implementation, this would map to a memory-mapped file or similar
        self.data_size = 10000 
        
    def __len__(self) -> int:
        return self.data_size

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a single example for pre-training (input_ids and labels).
        Labels are shifted by 1 relative to input_ids for autoregressive training.
        """
        # Mock data generation
        input_ids = torch.randint(0, 50000, (self.seq_len,), dtype=torch.long)
        labels = input_ids.clone()
        return {
            "input_ids": input_ids,
            "labels": labels
        }

def get_pretraining_dataloader(
    data_path: str, 
    batch_size: int, 
    seq_len: int = 2048, 
    num_workers: int = 4
) -> DataLoader:
    """
    Creates a DataLoader for pre-training.
    """
    dataset = PreTrainingDataset(data_path, seq_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
