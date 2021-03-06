import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import os
import pickle
import pandas as pd
from torch.utils.data import Dataset


class Dataset3D(Dataset):
    def __init__(self, names_file, root_dir, transform=torch.from_numpy):
        self.names_file = pd.read_csv(names_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.names_file)

    def __getitem__(self, idx):
        bsu_path = os.path.join(self.root_dir, self.names_file.iloc[idx, 0])
        return self.get_repre(bsu_path)
        
    def get_repre(self, bsu_path):
        # Load pair representation
        with open(bsu_path, 'rb') as f:
            representation = pickle.load(f)
        
        # Transform the representation (default: torch.from_numpy)
        if self.transform:
            representation = self.transform(representation)

        # Add one channel dimension
        representation = representation.unsqueeze(3)
        representation = representation.permute(3, 0, 1, 2)
        return representation
    
    
class Dataset2D(Dataset):
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
        # Load pair representation
        with open(bsu_path, 'rb') as f:
            representation = np.load(f, allow_pickle=True)

        # Resize the representation
        if self.transform:
            representation = self.transform(representation)
        return representation