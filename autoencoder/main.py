from numpy import True_
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
import os
import pickle
import pandas as pd
from models.model_1 import Conv2DAutoEncoder
from models.model_2 import Conv2DAutoEncoder
from torch.utils.data import (
    Dataset,
    DataLoader,
)  # PyTorch's data loading module
from data.dataset import BSUDataset


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 1
num_epochs = 20
learning_rate = 0.005

# Load data
dataset = BSUDataset(
    names_file = "bsu_names.csv",
    root_dir = "/scratch/hgao53/af2_research_model/bsu_padded/",
    transform = transforms.ToTensor(),
)

dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

# Model
model = Conv2DAutoEncoder(in_channel=128).to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print(len(dataset))

# Train model
for epoch in range(num_epochs):
    total_loss = 0

    for data in enumerate(dataloader):
        # Get data to cuda if possible
        i, bsu = data
        bsu = bsu.to(device=device)

        # ==================forward==================
        _, decoded = model(bsu)
        loss = criterion(bsu, decoded)

        # ==================backward==================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.data
    
        # ==================log==================
        print('epoch [{}/{}] -- batch [{}] loss:{:.4f}'
            .format(epoch+1, num_epochs, i, total_loss))
    
    torch.save(model.state_dict(), './conv2d_autoencoder_{0}.pth'.format(epoch))

# torch.save(model.state_dict(), './conv2d_autoencoder.pth')