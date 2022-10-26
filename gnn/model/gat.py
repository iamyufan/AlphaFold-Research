import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GATConv
import dgl.function as fn
import numpy as np


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
        return self.leakyrelu(self.batchnorm(self.conv(x)))


class GAT(nn.Module):
    def __init__(self,
                 g,
                 m_fea_dim,
                 num_hidden,
                 num_layers,
                 num_labels,
                 dropout,
                 device,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,):
        super(GAT, self).__init__()
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
        self.fc_list = nn.ModuleList(
            [nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)

        # GAT layers
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            num_hidden, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        
        # hidden layers
        for l in range(1, num_layers-1):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, False, self.activation))
            
        # output projection
        self.gat_layers.append(GATConv(
            num_hidden * heads[-2], num_labels, heads[-1],
            feat_drop, attn_drop, negative_slope, False, None))
        

    def forward(self, node_count_by_type, blocks, features_list):
        nodes_to_train = blocks[0].srcdata['_ID'].tolist()

        e_nodes = [n for n in nodes_to_train if n < node_count_by_type[0]]
        m_nodes = [n for n in nodes_to_train if n >= node_count_by_type[0]]

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

        e_feature_logits = e_feature_logits.to(self.device).permute(0, 3, 1, 2)
        for i, layer in enumerate(self.conv2d_layers):
            e_feature_logits = self.dropout_conv2d(e_feature_logits)
            e_feature_logits = layer(e_feature_logits)
        # After: e_feature_logits: [#e_nodes, 64]
        # print(f'e_feature_logits: {e_feature_logits.shape}')

        # 1.2 single representation vector
        e_feature_single = features_list[0][e_nodes]
        e_feature_single = self.single_linear(e_feature_single)
        # print(f'e_feature_single: {e_feature_single.shape}')

        ## 1.3 concatenate logits and single
        e_feature = torch.cat((e_feature_logits, e_feature_single), 1)
        e_dim = e_feature.shape[1]
        # After: e_feature: [384, 128]
        # print(f'e_feature: {e_feature.shape}')


        # 2. Pad the features of all the nodes and make them the same dimension
        m_feature = features_list[1][[idx-node_count_by_type[0] for idx in m_nodes]]
        e_feature = torch.cat((e_feature, torch.zeros((e_feature.shape[0], self.m_fea_dim))), 1)  # [384, e_dim+129] i.e. [616, 257]
        # print(f'e_feature: {e_feature.shape}')
        m_feature = torch.cat((torch.zeros((m_feature.shape[0], e_dim)), m_feature), 1)  # [616, 128+m_fea_dim] i.e. [616, 257]
        # print(f'm_feature: {m_feature.shape}')
        features_list = [e_feature, m_feature]
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0) # [1000, hidden_dim]
        # print(f'h: {h.shape}')


        # 3. GAT layers
        for l in range(self.num_layers-1):
            h = self.gat_layers[l](blocks[l], h).flatten(1)
        # output projection
        logits = self.gat_layers[-1](blocks[-1], h).mean(1)

        return logits


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

        e_feature_logits = e_feature_logits.permute(0, 3, 1, 2)
        for i, layer in enumerate(self.conv2d_layers):
            e_feature_logits = self.dropout_conv2d(e_feature_logits)
            e_feature_logits = layer(e_feature_logits)
        # After: e_feature_logits: [#e_nodes, 64]
        # print(f'e_feature_logits: {e_feature_logits.shape}')

        return e_feature_logits


    def inference(self, node_count_by_type, features_list):
        # 1. Enzyme feature
        ## 1.1 logits tensor (input [384, 1600, 1600, 2])]))
        e_feature_logits = None
        for pointer in range(node_count_by_type[0] // 100 + 1):
            nodes_to_infer = list(range(pointer*100, min(pointer*100+100, node_count_by_type[0])))
            e_feature_logits_t = self.get_e_feature_logits(node_count_by_type, nodes_to_infer)

            if e_feature_logits == None:
                e_feature_logits = e_feature_logits_t
            else:
                e_feature_logits = torch.cat([e_feature_logits, e_feature_logits_t])

        ## 1.2 single representation vector
        e_feature_single = features_list[0]
        e_feature_single = self.single_linear(e_feature_single)
        # After: e_feature_single: [384, 64]
        # print(f'e_feature_single: {e_feature_single.shape}')

        ## 1.3 concatenate logits and single
        e_feature = torch.cat((e_feature_logits, e_feature_single), 1)
        e_dim = e_feature.shape[1]
        # After: e_feature: [384, 128]
        # print(f'e_feature: {e_feature.shape}')

        # 2. Pad the features of all the nodes and make them the same dimension
        m_feature = features_list[1]
        
        e_feature = torch.cat((e_feature, torch.zeros((e_feature.shape[0], self.m_fea_dim))), 1)  # [384, e_dim+129] i.e. [616, 257]
        # print(f'e_feature: {e_feature.shape}')
        
        m_feature = torch.cat((torch.zeros((m_feature.shape[0], e_dim)), m_feature), 1)  # [616, 128+m_fea_dim] i.e. [616, 257]
        # print(f'm_feature: {m_feature.shape}')
        
        features_list = [e_feature, m_feature]
        
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0) # [1000, hidden_dim]
        
        # print(f'h: {h.shape}')
        
        # 3. GAT layers
        for l in range(self.num_layers-1):
            h = self.gat_layers[l](self.g, h).flatten(1)
        # output projection
        logits = self.gat_layers[-1](self.g, h).mean(1)
            
        # After: h: [1000, num_labels]

        return logits