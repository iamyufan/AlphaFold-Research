import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import os
import pickle
import pandas as pd
from torch.utils.data import Dataset


class AF2OutputDataset(Dataset):
    def __init__(self, names_file, root_dir, transform=None):
        self.names_file = pd.read_csv(names_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.names_file)

    def __getitem__(self, idx):
        bsu_path = os.path.join(self.root_dir, self.names_file.iloc[idx, 0])
        return self.get_repre(bsu_path)
        
    def get_repre(self, bsu_path):
        # Read bsu and extract representation
        with open(bsu_path, 'rb') as f:
            representation = pickle.load(f)
        
        if self.transform:
            representation = self.transform(representation)

        representation = representation.unsqueeze(3)

        representation = representation.permute(3, 0, 1, 2)

        return representation