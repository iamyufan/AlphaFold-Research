import torch
import torch.nn as nn
import dgl
from dgl.nn.pytorch import GraphConv
import dgl.function as fn


# (out_channels, kernel_size, stride, padding)
enzyme_conv_archi = [
    # input: 1600x1600x1
    (1, 4, 4, 0),
    # 400x400x1
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
        self.layers = nn.ModuleList()
        
        # conv2d layers for enzyme nodes
        e_layers = []
        in_channels = 1
        for x in enzyme_conv_archi:
            if type(x) == tuple:
                e_layers += [
                    Conv2dBlock(
                        in_channels, x[0], kernel_size=x[1], stride=x[2], padding=x[3],
                    )
                ]
                in_channels = x[0]
                
        self.e_conv = nn.Sequential(*e_layers)

        # fc layers: to make the features of all the nodes become the same dimension  
        in_dims = [25*25, m_dim]
              
        self.fc_list = nn.ModuleList(
            [nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)

        # input layer
        self.layers.append(
            GraphConv(num_hidden, num_hidden, activation=activation))

        # hidden layers
        for i in range(num_layers - 1):
            self.layers.append(
                GraphConv(num_hidden, num_hidden, activation=activation))

        # output layer
        self.layers.append(GraphConv(num_hidden, num_labels))
        self.dropout = nn.Dropout(p=dropout)


    def forward(self, features_list):
        h = []
        
        e_feature = features_list[0].unsqueeze(1)
        e_feature = self.dropout(e_feature)
        e_feature = self.e_conv(e_feature).reshape((-1, 625))
        
        m_feature = features_list[1]
        
        features_list = [e_feature, m_feature]
        
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0)
        for i, layer in enumerate(self.layers):
            h = self.dropout(h)
            h = layer(self.g, h)
        return h
