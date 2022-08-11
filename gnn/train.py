import time
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import dgl
from utils import EarlyStopping
from sklearn.metrics import r2_score
# data loading modules
from data.load_data import load_data
# gnn models
from model.gcn import GCN


def main(args):
    # Set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('>> device: {}'.format(device))

    # Load data
    dataset = args.dataset
    data = load_data(dataset, device)  # data is a dict
    features_list = data['features_list']   # list of features
    num_labels = data['num_labels']         # number of labels
    m_fea_dim = data['m_feature_dim']       # molecular feature dimension
    g = data['g']                           # dgl graph
    ## train, val, test
    train_idx = data['train_val_test_idx']['train_idx']
    val_idx = data['train_val_test_idx']['val_idx']
    test_idx = data['train_val_test_idx']['test_idx']
    labels = data['labels'].unsqueeze(-1)   # labels
    node_count_by_type = dict(data['rd'].nodes['count'])

    # Create model
    net = GCN(dataset=dataset,
            g=g, 
            m_fea_dim=m_fea_dim, 
            num_hidden=args.hidden_dim, 
            num_labels=num_labels, 
            num_layers=args.num_layers,
            dropout=args.dropout,
            device=device)
    net.to(device)

    # Set loss and optimizer
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(
        net.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )

    # Set early stopping
    patience = 5
    counter = 0
    # early_stopping = EarlyStopping(patience, verbose=True)

    # Train
    best_val_r2 = -10
    best_test_r2 = -10
    print('------ Training ------')
    for epoch in range(args.epoch):
        t_start = time.time()
        net.train()
        
        # forward
        output_predictions = net(node_count_by_type, features_list)
        loss = criterion(output_predictions[train_idx], labels[train_idx])

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute R2 scores on training/validation/test
        net.eval()
        train_r2 = r2_score(labels[train_idx].cpu().numpy(), output_predictions[train_idx].cpu().detach().numpy())
        val_r2 = r2_score(labels[val_idx].cpu().numpy(), output_predictions[val_idx].cpu().detach().numpy())
        test_r2 = r2_score(labels[test_idx].cpu().numpy(), output_predictions[test_idx].cpu().detach().numpy())

        t_end = time.time()

        # Log
        print('Epoch {:02d} | Loss: {:.4f} | Time: {:.4f}'.format(
            epoch, loss, t_end-t_start))
        print('Epoch {:02d} | Train R2: {:.4f} | Val R2: {:.4f} (Best {:.4f}) | Test R2: {:.4f} (Best {:.4f})'.format(
            epoch, train_r2, val_r2, best_val_r2, test_r2, best_test_r2))

        # Save the best model on validation set
        if best_val_r2 < val_r2:
            best_val_r2 = val_r2
            best_test_r2 = test_r2
            torch.save(net.state_dict(),
                   'checkpoints/checkpoint_{}_{}.pth'.format(dataset, args.model_type))
        else:
            counter += 1
            if counter == patience:
                print('Early Stopping!')
                break
            else:
                print(f'Early Stopping Counter: {counter}/{patience}')

    # Test model
    net.load_state_dict(torch.load('checkpoints/checkpoint_{}_{}.pth'.format(dataset, args.model_type)))
    net.eval()
    with torch.no_grad():
        output_labels = labels[test_idx].cpu().numpy()
        output_predictions = net(node_count_by_type, features_list)
        output_predictions = output_predictions[test_idx].cpu().numpy()
        km_r2 = r2_score(output_labels, output_predictions)
        print('\nR2 score for km: {}'.format(km_r2))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='GNN')

    # Data options
    parser.add_argument('--dataset', type=str, default='iYO844')

    # Model options
    parser.add_argument('--model-type', type=str,
                        default='gcn', help="gcn or gat")
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--slope', type=float, default=0.05)

    # Training options
    parser.add_argument('--hidden-dim', type=int, default=96,
                        help='Dimension of the node hidden state. Default is 128.')
    parser.add_argument('--num-heads', type=int, default=8,
                        help='Number of the attention heads. Default is 8.')
    parser.add_argument('--epoch', type=int, default=5,
                        help='Number of epochs.')
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--weight-decay', type=float, default=1e-4)

    args = parser.parse_args()

    print(args)
    main(args)
