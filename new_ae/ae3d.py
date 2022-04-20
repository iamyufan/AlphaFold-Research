from numpy import True_
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
import os
import pickle
import pandas as pd
from models.conv_ae_3d import Conv3DAutoEncoder
from torch.utils.data import DataLoader 
from data.dataset3d import AF2OutputDataset


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 2
num_epochs = 20
learning_rate = 0.01

# Load data
dataset = AF2OutputDataset(
    names_file = "n.csv",
    root_dir = "/scratch/hgao53/padded_bsu",
    transform = transforms.ToTensor(),
)

dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

# Model
model = Conv3DAutoEncoder(in_channel=1).to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print(len(dataset))

print('================= Begin Training =================')

# Train model
for epoch in range(num_epochs):
    total_loss = 0

    epoch_time_start=time.time()
    for data in enumerate(dataloader):
        batch_time_start=time.time()
        
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
        
        batch_time_end=time.time()
    
        # ==================log==================
        print('=== epoch [{}/{}] -- batch [{}] -- time:{} -- loss:{:.4f}'
              .format(epoch, num_epochs, i, batch_time_end-batch_time_start, total_loss))
    
    epoch_time_end=time.time()
    print('=== epoch [{}/{}] -- time:{}\n'.format(epoch, num_epochs, epoch_time_end-epoch_time_start))
    torch.save(model.state_dict(), './conv_ae_3d_{0}.pth'.format(epoch))