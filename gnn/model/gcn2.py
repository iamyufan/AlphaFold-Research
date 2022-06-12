import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GraphConv
import dgl.function as fn
import numpy as np
from scipy.linalg import fractional_matrix_power


# (out_channels, kernel_size, stride, padding)
enzyme_conv_archi = [
    # input: 1600x1600x2
    (2, 4, 4, 0),
    # 400x400x2
    (1, 4, 4, 0),
    # 100x100x1
    (1, 4, 4, 0),
    # 25x25x1
]


class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs): 
        super(Conv2dBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        # print(x.dtype)
        return self.leakyrelu(self.batchnorm(self.conv(x)))


class GCN(nn.Module):
    def __init__(self,
                 g,
                 m_fea_dim,
                 num_hidden,
                 num_layers,
                 num_labels,
                 dropout,
                 device):
        super(GCN, self).__init__()
        self.g = g
        self.m_fea_dim = m_fea_dim
        self.device = device

        # Enzyme feature
        ## Conv2d layers for enzyme logits
        self.conv2d_layers = nn.ModuleList()

        in_channels = 2
        for x in enzyme_conv_archi:
            self.conv2d_layers.append(Conv2dBlock(in_channels, x[0], kernel_size=x[1], stride=x[2], padding=x[3],))
            in_channels = x[0]
            
        self.conv2d_layers.append(nn.Flatten(start_dim=1))
        self.conv2d_layers.append(nn.Linear(625, 64))
        self.dropout_conv2d = nn.Dropout(dropout)

        ## Linear layers for enzyme single
        self.single_linear = nn.Linear(1600, 64)

        # fc layers: to make the features of all the nodes become the same dimension  
        in_dims = [257, 257]

        # GC layers 
        self.GClayers = nn.ModuleList()
        ## input layer
        self.GClayers.append(GraphConv(257, num_hidden, activation=F.leaky_relu))
        for i in range(num_layers-2):
            self.GClayers.append(
                GraphConv(num_hidden, num_hidden, activation=F.leaky_relu))
        ## output layer
        self.GClayers.append(GraphConv(num_hidden, num_labels))
        self.dropout_GC = nn.Dropout(p=dropout)


    def get_e_feature_logits(self, node_count_by_type, nodes_to_infer):
        e_nodes = [n for n in nodes_to_infer if n < node_count_by_type[0]]

        # 1. Enzyme feature
        ## 1.1 logits tensor (input [#e_nodes, 1600, 1600, 2])]))
        e_feature_logits = None
        for e_node in e_nodes:
            with open('../datasets/iYO844/logits/{}.npy'.format(e_node), 'rb') as f:
                logits = torch.unsqueeze(torch.from_numpy(np.load(f)), 0)
            if e_feature_logits == None:
                e_feature_logits = logits
            else:
                e_feature_logits = torch.cat([e_feature_logits, logits])

        e_feature_logits = e_feature_logits.permute(0, 3, 1, 2).to(self.device)
        for i, layer in enumerate(self.conv2d_layers):
            e_feature_logits = self.dropout_conv2d(e_feature_logits)
            e_feature_logits = layer(e_feature_logits)
        # After: e_feature_logits: [#e_nodes, 64]
        # print(f'e_feature_logits: {e_feature_logits.shape}')

        return e_feature_logits


    def forward(self, node_count_by_type, features_list):
        # 1. Enzyme feature
        ## 1.1 logits tensor (input [384, 1600, 1600, 2])]))
        e_feature_logits = None
        for pointer in range(node_count_by_type[0] // 50 + 1):
            nodes_to_infer = list(range(pointer*50, min(pointer*50+50, node_count_by_type[0])))
            e_nodes = [n for n in nodes_to_infer if n < node_count_by_type[0]]
            e_feature_logits_t = None
            for e_node in e_nodes:
                with open('../datasets/iYO844/logits/{}.npy'.format(e_node), 'rb') as f:
                    logits = torch.unsqueeze(torch.from_numpy(np.load(f)), 0)
                if e_feature_logits_t == None:
                    e_feature_logits_t = logits
                else:
                    e_feature_logits_t = torch.cat([e_feature_logits_t, logits])

            e_feature_logits_t = e_feature_logits_t.permute(0, 3, 1, 2).to(self.device)
            for i, layer in enumerate(self.conv2d_layers):
                # e_feature_logits_t = self.dropout_conv2d(e_feature_logits_t)
                e_feature_logits_t = layer(e_feature_logits_t)

            # e_feature_logits_t = self.get_e_feature_logits(node_count_by_type, nodes_to_infer)

            if e_feature_logits == None:
                e_feature_logits = e_feature_logits_t
            else:
                e_feature_logits = torch.cat([e_feature_logits, e_feature_logits_t])

        e_feature_logits = F.normalize(e_feature_logits)

        ## 1.2 single representation vector
        e_feature_single = features_list[0]
        e_feature_single = self.single_linear(e_feature_single)
        e_feature_single = F.normalize(e_feature_single)

        # After: e_feature_single: [384, 64]
        # print(f'e_feature_single: {e_feature_single.shape}')

        ## 1.3 concatenate logits and single
        e_feature = torch.cat((e_feature_logits, e_feature_single), 1)
        e_dim = e_feature.shape[1]
        # After: e_feature: [384, 128]
        # print(f'e_feature: {e_feature.shape}')

        # 2. Pad the features of all the nodes and make them the same dimension
        m_feature = features_list[1]
        e_feature = torch.cat((e_feature, torch.zeros((e_feature.shape[0], self.m_fea_dim)).to(self.device)), 1)  # [384, e_dim+129] i.e. [384, 257]
        m_feature = torch.cat((torch.zeros((m_feature.shape[0], e_dim)).to(self.device), m_feature), 1)  # [616, 128+m_fea_dim] i.e. [616, 257]
        features_list = [e_feature, m_feature]

        h = torch.cat(features_list, 0) # [1000, 257]

        # 2.2 Aggregate node feature
        A = torch.tensor(self.g.adj(scipy_fmt='coo').todense(), dtype=torch.float).to(self.device)
        degrees = self.g.in_degrees()
        D = torch.zeros((degrees.shape[0]), degrees.shape[0])
        for i in range(degrees.shape[0]):
            D[i][i] = degrees[i]
        D_half_norm = torch.tensor(fractional_matrix_power(D, -0.5), dtype=torch.float).to(self.device)
        h = D_half_norm.matmul(A).matmul(D_half_norm).matmul(h).to(self.device)
        
        # 3. GC layers
        for i, layer in enumerate(self.GClayers):
            h = self.dropout_GC(h)
            h = layer(self.g, h)
            
        # After: h: [1000, num_labels]

        return h