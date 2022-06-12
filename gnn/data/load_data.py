import numpy as np
import torch
from data.reaction_dataset import ReactionDataset
import dgl


def sp_to_spt(mat):
    coo = mat.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))

def mat2tensor(mat):
    if type(mat) is dict:
        result = dict()
        for key in mat:
            result[key] = torch.from_numpy(mat[key]).type(torch.FloatTensor)
        return result
    if type(mat) is np.ndarray:
        return torch.from_numpy(mat).type(torch.FloatTensor)
    return sp_to_spt(mat)

def load_data(model_name, device):
    data = dict()

    rd = ReactionDataset(f'../datasets/{model_name}')
    print('>> Dataset created')

    data['rd'] = rd

    # features_list: features of the nodes of each node_type_id
    features_list = list()
    for i in range(len(rd.nodes['count'])):
        features = rd.nodes['attr'][i]
        features = mat2tensor(features).to(device)
        features_list.append(features)
    data['features_list'] = features_list
    print('>> feature_list created')

    # adjM: adjacancy matrix of the graph
    adjM = sum(rd.links['data'].values())
    data['adjM'] = adjM
    print('>> adjM created')

    # train_val_test_idx
    np.random.seed(1)
    val_ratio = 0.3
    train_idx = np.nonzero(rd.labels_train['mask'])[0]
    np.random.shuffle(train_idx)
    split = int(train_idx.shape[0]*val_ratio)

    val_idx = np.sort(train_idx[:split])
    train_idx = np.sort(train_idx[split:])
    test_idx = np.sort(np.nonzero(rd.labels_test['mask'])[0])
    
    train_val_test_idx = {}
    train_val_test_idx['train_idx'] = train_idx
    train_val_test_idx['val_idx'] = val_idx
    train_val_test_idx['test_idx'] = test_idx
    data['train_val_test_idx'] = train_val_test_idx
    print('>> train_val_test_idx set')

    # labels
    labels = np.zeros((rd.nodes['count'][0],), dtype=float)
    labels[train_idx] = rd.labels_train['data'][train_idx]
    labels[val_idx] = rd.labels_train['data'][val_idx]
    labels[test_idx] = rd.labels_test['data'][test_idx]
    labels = torch.FloatTensor(labels).to(device)
    data['labels'] = labels
    print('>> labels set')

    # g: graph
    g = dgl.from_scipy(adjM+(adjM.T))
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    g = g.to(device)
    data['g'] = g
    print('>> Graph built')

    # num_label
    num_labels = rd.labels_train['num_labels']
    data['num_labels'] = num_labels

    # m_fea_dim
    data['m_feature_dim'] = features_list[1].shape[1]

    return data