from numpy import True_
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from models.conv_ae_2d import Conv2DAutoEncoder
from models.conv_ae_3d import Conv3DAutoEncoder
from torch.utils.data import DataLoader 
from dataset import Dataset2D, Dataset3D

def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate

    # Load data
    root_dir = args.root_dir
    if args.dataset == '3d':
        dataset = Dataset3D(
            names_file = "filename3d.csv",
            root_dir = root_dir,
            transform = torch.from_numpy,
        )
    elif args.dataset == '2d':
        dataset = Dataset2D(
            names_file = "filename2d.csv",
            root_dir = root_dir,
            transform = torch.from_numpy,
        )

    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    # Model
    if args.model == 'conv_ae_3d':
        model = Conv3DAutoEncoder(in_channel=1).to(device)
    elif args.model == 'conv_ae_2d':
        model = Conv2DAutoEncoder(in_channel=1).to(device)

    PATH = 'conv_ae_3d_1.pth'
    model.load_state_dict(torch.load(PATH))
    
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
        print('=== epoch [{}/{}] -- time:{}\n'.format(epoch+2, num_epochs+2, epoch_time_end-epoch_time_start))
        torch.save(model.state_dict(), './conv_ae_3d_{0}.pth'.format(epoch+2))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description = 'AutoEncoder')
    # Dataset
    parser.add_argument('-d', '--dataset', type=str, default='3d', help='Dataset')
    parser.add_argument('-dir', '--root-dir', type=str, default='"/scratch/hgao53/padded_bsu"',
                        help='Dir for tensor-like data')
    parser.add_argument('-m', '--model', type=str, default='conv_ae_3d',)
    
    # Training hyper-parameters
    parser.add_argument('-e', '--num-epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('-b', '--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.01, help='Learning rate')
    
    args = parser.parse_args()
    print(args)
    main(args)