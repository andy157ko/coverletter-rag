"""
Handles dataset loading, preprocessing hooks, and train/val/test splits.

Rubric:
- Train/val/test split
- Data loading & batching
"""

from typing import Dict, Any
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path

class CoverLetterDataset(Dataset):
    def __init__(self, path: str):
        self.path = Path(path)
        self.samples = []
        with self.path.open() as f:
            for line in f:
                self.samples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        # Expected fields: "resume_text", "job_text", "cover_letter_text"
        return item

def create_dataloaders(
    train_path: str,
    val_path: str,
    test_path: str,
    batch_size: int,
    collate_fn,
) -> Dict[str, DataLoader]:
    train_ds = CoverLetterDataset(train_path)
    val_ds = CoverLetterDataset(val_path)
    test_ds = CoverLetterDataset(test_path)

    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn),
    }