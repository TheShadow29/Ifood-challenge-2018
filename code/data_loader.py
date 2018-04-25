import torch
from torch.utils.data import Dataset


class INat_simple(Dataset):
    def __init__(self, config):
        super().__init__()
        self.config = config
