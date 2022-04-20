import sys
sys.path.append('../../')  # gnn
import time
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import dgl

def main(args):
    # Load data
    
    # Set device
    
    # Set model
    ## GCN
    ## GAT
    ## HAN
    ## GTN
    
    # Set loss and optimizer
    
    # Train model
    for epoch in range(args.epoch):
        # ==================forward==================
        
        # ==================backward=================
        
        # ====================log====================
        print('1')
    # Test model
    

if __name__ == '__main__':
    import argparse

    from utils import setup

    parser = argparse.ArgumentParser('HAN')
    parser.add_argument('-s', '--seed', type=int, default=1,
                        help='Random seed')
    parser.add_argument('-ld', '--log-dir', type=str, default='results',
                        help='Dir for saving training results')
    parser.add_argument('--hetero', action='store_true',
                        help='Use metapath coalescing with DGL\'s own dataset')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dataset', type=str, default='DBLP',
                        choices=['DBLP', 'ACM', 'Freebase'])
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=100)
    args = parser.parse_args().__dict__

    args = setup(args)
    print(args)
    main(args)