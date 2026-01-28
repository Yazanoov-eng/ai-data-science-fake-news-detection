import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class NewsDataset(Dataset):
    def __init__(self, csv_path: str, tokenizer_name: str, max_length: int):
        self.df = pd.read_csv(csv_path)
        self.texts = self.df["content"].astype(str).tolist()
        self.labels = self.df["label"].astype(int).tolist()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
