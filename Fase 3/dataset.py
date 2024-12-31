import torch
from torch.utils.data import Dataset
from preprocessing import *


class CodeDataset(Dataset):
    def __init__(self, codes, vocab):
        self.data = [tokens_to_indices(tokenize_code(preprocess_code(code)), vocab) for code in codes]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.long)


# Función para dividir el dataset
def split_dataset(dataset, train_ratio=0.8):
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    return torch.utils.data.random_split(dataset, [train_size, val_size])
