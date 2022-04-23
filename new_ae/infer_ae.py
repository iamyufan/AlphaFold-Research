from numpy import True_
import numpy as np
from numpy import save
import time
import torch
import torch.nn as nn
import torch.optim as optim
from models.conv_ae_2d import Conv2DAutoEncoder
from models.conv_ae_3d import Conv3DAutoEncoder
from torch.utils.data import DataLoader 
from dataset import Dataset2D, Dataset3D
import os
import pickle

def get_files(directory):
    return [os.path.join(directory, f) for f in sorted(list(os.listdir(directory)))
            if os.path.isfile(os.path.join(directory, f))]

def main():
    # Set device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = Conv3DAutoEncoder(in_channel=1).to('cpu')
    PATH = 'conv_ae_3d_1.pth'
    model.load_state_dict(torch.load(PATH))
    # model = model.double()
    
    dataset = 'imm904'
    if dataset == 'iqo884':                                             # BSU
        root_dir = "/scratch/hgao53/padded_bsu"
    elif dataset == 'imm904':                                           # YAL
        root_dir = '/home/hgao53/alphafold_new/alphafold/final'
    elif dataset == 'iml1515':                                          # b
        root_dir = '/home/hgao53/alphafold_new/alphafold/final_904'
    # SAVE_DIR = '/scratch/hgao53/encoded_bsu'
    
    output = dict()
    
    file_names = get_files(root_dir)
    for i, item in enumerate(file_names):
        print('Processing %i of %i (%s)' % (i+1, len(file_names), item))
        with open(item, 'rb') as f:
            data = pickle.load(f)
            
        if dataset != 'iqo884':
            data = data['representations']['pair']
            data = np.pad(data, ((0, 2048-data.shape[0]), (0, 2048-data.shape[0]), (0, 0)), 'constant')
        
        transform = torch.from_numpy
        data = transform(data)
        
        # Add channel dimension
        data = data.unsqueeze(3)
        data = data.permute(3, 0, 1, 2)
        
        data = data.unsqueeze(4)
        data = data.permute(4, 0, 1, 2, 3)
        data.to('cpu')#.double()
        
        # Infer
        encoded, _ = model(data)
        encoded = encoded.squeeze(0).squeeze(0).squeeze(2).reshape(-1).detach().numpy()
        
        output[item] = encoded
        # SAVE_PATH = SAVE_DIR + item + 'npy'
        # save(SAVE_PATH, encoded)
        
        # Log
        print('Completed {}/{}'.format(i+1, len(file_names)))
    
    with open('{}_output.pkl'.format(dataset), 'wb') as f:
        pickle.dump(output, f)
    print('Completed')


if __name__ == '__main__':
    main()