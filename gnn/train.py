import sys
import time
import torch
import torch.nn.functional as F
from utils.utils import load_data, mat2tensor, regression_loss
from model.gcn import GCN
from model.gat import GAT
import numpy as np
import dgl


def main(args):
    # Set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load data
    features_list, adjM, labels, train_val_test_idx, dl = load_data(args.dataset)
    features_list = [mat2tensor(features).to(device)
                     for features in features_list]
    m_dim = features_list[1].shape[1]

    # Set train, val, test index
    labels = torch.FloatTensor(labels).to(device)
    train_idx = train_val_test_idx['train_idx']
    train_idx = np.sort(train_idx)
    val_idx = train_val_test_idx['val_idx']
    val_idx = np.sort(val_idx)
    test_idx = train_val_test_idx['test_idx']
    test_idx = np.sort(test_idx)

    # Build graph
    g = dgl.from_scipy(adjM+(adjM.T))
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    g = g.to(device)

    # Set model
    num_labels = dl.labels_train['num_labels']

    ## GAT
    if args.model_type == 'gat':
        heads = [args.num_heads] * args.num_layers + [1]
        net = GAT(g, m_dim, args.hidden_dim, num_labels, args.num_layers,
                  heads, F.elu, args.dropout, args.dropout, args.slope, False)
    ## GCN
    elif args.model_type == 'gcn':
        net = GCN(g, m_dim, args.hidden_dim, num_labels,
                  args.num_layers, F.elu, args.dropout)
    ## HAN
    ## GTN

    net.to(device)
    
    print(net)

    # Set loss and optimizer
    optimizer = torch.optim.Adam(
        net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Train model
    for epoch in range(args.epoch):
        net.train()
        t_start = time.time()
        # ==================forward==================
        net.train()
        logits = net(features_list)

        train_loss = regression_loss(logits[train_idx], labels[train_idx])

        # ==================backward=================
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        t_end = time.time()

        # ====================log====================
        print('Epoch {:05d} | Train_Loss: {:.4f} | Time: {:.4f}'.format(
            epoch, train_loss.item(), t_end-t_start))

        # ====================validatation====================
        t_start = time.time()
        net.eval()
        with torch.no_grad():
            logits = net(features_list)
            val_loss = regression_loss(logits[val_idx], labels[val_idx])
        t_end = time.time()
        # print validation info
        print('Epoch {:05d} | Val_Loss {:.4f} | Time(s) {:.4f}'.format(
            epoch, val_loss.item(), t_end - t_start))
        print()
        torch.save(net.state_dict(),
                   'checkpoint/checkpoint_{}_{}_{}.pth'.format(args.dataset, args.model_type, epoch))

    # Test model
    net.load_state_dict(torch.load(
        'checkpoint/checkpoint_{}_{}_{}.pth'.format(args.dataset, args.model_type, args.epoch-1)))
    net.eval()
    test_logits = []
    with torch.no_grad():
        logits = net(features_list)
        test_logits = logits[test_idx]
        # dl.gen_file_for_evaluate(test_idx=test_idx,label=pred)
        pred = test_logits.cpu().numpy()
        print(dl.evaluate(pred))


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
    parser.add_argument('--hidden-dim', type=int, default=64,
                        help='Dimension of the node hidden state. Default is 64.')
    parser.add_argument('--num-heads', type=int, default=8,
                        help='Number of the attention heads. Default is 8.')
    parser.add_argument('--epoch', type=int, default=10,
                        help='Number of epochs.')
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--weight-decay', type=float, default=1e-4)

    args = parser.parse_args()

    print(args)
    main(args)
