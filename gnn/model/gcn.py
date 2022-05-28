import torch
import torch.nn as nn
import dgl
from dgl.nn.pytorch import GraphConv
import dgl.function as fn


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


class GCN(nn.Module):
    def __init__(self,
                 g,
                 m_dim,
                 num_hidden,
                 num_labels,
                 num_layers,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.g = g
        
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
        
        # GC layers 
        self.GClayers = nn.ModuleList() 
        ## input layer
        self.GClayers.append(
            GraphConv(num_hidden, num_hidden, activation=activation))

        ## hidden layers
        for i in range(num_layers - 1):
            self.GClayers.append(
                GraphConv(num_hidden, num_hidden, activation=activation))

        ## output layer
        self.GClayers.append(GraphConv(num_hidden, num_labels))
        self.dropout_GC = nn.Dropout(p=dropout)

    def forward(self, features_list):        
        # 1. Enzyme feature
        ## 1.1 logits tensor (input [384, 1600, 1600, 2])]))
        e_feature_logits = features_list[0]['logits'].permute(0, 3, 1, 2)

        for i, layer in enumerate(self.conv2d_layers):
            e_feature_logits = self.dropout_conv2d(e_feature_logits)
            e_feature_logits = layer(e_feature_logits)
        # After: e_feature_logits: [384, 64]
        # print(f'e_feature_logits: {e_feature_logits.shape}')
        
        ## 1.2 single representation vector
        e_feature_single = features_list[0]['single']
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
        
        e_feature = torch.cat((e_feature, torch.zeros((e_feature.shape[0], m_dim))), 1)  # [384, e_dim+129] i.e. [616, 257]
        # print(f'e_feature: {e_feature.shape}')
        
        m_feature = torch.cat((torch.zeros((m_feature.shape[0], e_dim)), m_feature), 1)  # [616, 128+m_dim] i.e. [616, 257]
        # print(f'm_feature: {m_feature.shape}')
        
        features_list = [e_feature, m_feature]
        
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0) # [1000, hidden_dim]
        
        # print(f'h: {h.shape}')
        
        # 3. GC layers
        for i, layer in enumerate(self.GClayers):
            h = self.dropout_GC(h)
            h = layer(self.g, h)
            
        # After: h: [1000, num_labels]

        return h
