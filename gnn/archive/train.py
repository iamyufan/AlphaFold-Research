import time
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import dgl
# data loading modules
from data.reaction_dataset import ReactionDataset
from data.load_data import load_data
# gnn models
from model.gcn import GCN
from model.gat import GAT
# utils
from utils import regression_loss, EarlyStopping
from sklearn.metrics import r2_score


def main(args):
    # Set device
    device = 'cpu' #torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    # Load data
    data = load_data(args.dataset, device)

    features_list = data['features_list']
    num_labels = data['num_labels']
    m_fea_dim = data['m_feature_dim']

    g = data['g']

    train_idx = data['train_val_test_idx']['train_idx']
    val_idx = data['train_val_test_idx']['val_idx']
    test_idx = data['train_val_test_idx']['test_idx']

    labels = data['labels']

    node_count_by_type = dict(data['rd'].nodes['count'])


    # Node data loader
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(args.num_layers)
    dataloader = dgl.dataloading.NodeDataLoader(
        g, train_idx, sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        device=device,
        num_workers=4)


    # Create model
    ## GAT
    if args.model_type == 'gat':
        heads = [args.num_heads] * args.num_layers + [1]
        net = GAT(g=g,
                 m_fea_dim=m_fea_dim,
                 num_hidden=args.hidden_dim,
                 num_layers=args.num_layers,
                 num_labels=num_labels,
                 dropout=args.dropout,
                 device=device,
                 heads=heads,
                 activation=F.elu,
                 feat_drop=args.dropout,
                 attn_drop=args.dropout,
                 negative_slope=args.slope)
    ## GCN
    elif args.model_type == 'gcn':
        net = GCN(g=g, 
                m_fea_dim=m_fea_dim, 
                num_hidden=args.hidden_dim, 
                num_labels=num_labels, 
                num_layers=args.num_layers,
                dropout=args.dropout,
                device=device)


    # Set loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), 
                                    lr=args.lr, 
                                    weight_decay=args.weight_decay)


    # Set early stopping
    patience = 3
    early_stopping = EarlyStopping(patience, verbose=True)


    # Train
    for epoch in range(args.epoch):
        t_start = time.time()
        epoch_loss = 0

        net.train()
    
        for input_nodes, output_nodes, blocks in dataloader:
            # ================== forward ==================
            blocks = [b.to(device) for b in blocks]

            output_labels = labels[output_nodes.tolist()]
            output_predictions = net(node_count_by_type, blocks, features_list)

            # ================== backward ==================
            loss = criterion(output_predictions.view(-1), output_labels)
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # ==================== log ====================
        t_end = time.time()

        print('Epoch {:05d} | Train_Loss: {:.4f} | Time: {:.4f}'.format(
            epoch, epoch_loss, t_end-t_start))

        torch.save(net.state_dict(),
                   'checkpoints/train/checkpoint_{}_{}_{}.pth'.format(args.dataset, args.model_type, epoch))

        # ==================== validatation ====================
        t_start = time.time()
        net.eval()
        with torch.no_grad():
            output_predictions = net.inference(node_count_by_type, features_list)
            val_loss = regression_loss(output_predictions[val_idx].view(-1), labels[val_idx])

        early_stopping(val_loss, net, model_type=args.model_type)
        t_end = time.time()

        print('Epoch {:05d} | Val_Loss {:.4f} | Time(s) {:.4f}'.format(
            epoch, val_loss.item(), t_end - t_start))
        print()

        if early_stopping.early_stop:
            print("> Early stopping!")
            break

    print('------ Full Training ------')
    patience = 3
    early_stopping = EarlyStopping(patience, verbose=True)

    net.load_state_dict(torch.load(f'checkpoint_{args.model_type}.pt'))
    
    for epoch in range(args.epoch):
        t_start = time.time()
        epoch_loss = 0

        net.train()

        output_labels = labels[train_idx]
        output_predictions = net.inference(node_count_by_type, features_list)

        loss = criterion(output_predictions[train_idx].view(-1), output_labels)

        epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        t_end = time.time()

        print('Epoch {:05d} {} | Train_Loss: {:.4f} | Time: {:.4f}'.format(
            epoch, 'full_training', epoch_loss, t_end-t_start))

        # ====================validatation====================
        t_start = time.time()
        net.eval()
        with torch.no_grad():
            logits = net.inference(node_count_by_type, features_list)
            val_loss = regression_loss(logits[val_idx].view(-1), labels[val_idx])
        t_end = time.time()
        # print validation info
        print('Epoch {:05d} {} | Val_Loss {:.4f} | Time(s) {:.4f}'.format(
            epoch, 'full_training', val_loss.item(), t_end - t_start))
        print()
        torch.save(net.state_dict(),
                   'checkpoints/train/checkpoint_{}_{}_{}.pth'.format(args.dataset, args.model_type, epoch+args.epoch))
        early_stopping(val_loss, net, model_type=args.model_type)
        if early_stopping.early_stop:
            print("> Early stopping!")
            last_epoch = epoch
            break


    # Test model
    net.load_state_dict(torch.load(f'checkpoint_{args.model_type}.pt'))
    net.eval()
    with torch.no_grad():
        output_labels = labels[test_idx].cpu().numpy()
        output_predictions = net.inference(node_count_by_type, features_list)
        output_predictions = output_predictions.view(-1)[test_idx].cpu().numpy()
        km_r2 = r2_score(output_labels, output_predictions)
        print('R2 score for km: {}'.format(km_r2))
        

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
    parser.add_argument('--batch-size', type=int, default=10)
    parser.add_argument('--hidden-dim', type=int, default=128,
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
