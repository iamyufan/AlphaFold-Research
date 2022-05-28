import imp
import numpy as np
import sys
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from utils.data_loader import data_loader
# from data_loader import data_loader


def load_data(prefix='iYO844'):
    dl = data_loader('./data/{}'.format(prefix))
    features = []
    for i in range(len(dl.nodes['count'])):
        th = dl.nodes['attr'][i]
        if th is None:
            features.append(sp.eye(dl.nodes['count'][i]))
        else:
            features.append(th)
    adjM = sum(dl.links['data'].values())
    # labels = np.zeros(
    #     (dl.nodes['count'][0], dl.labels_train['num_labels']), dtype=float)

    labels = np.zeros(
        (dl.nodes['count'][0],), dtype=float)
    val_ratio = 0.2
    train_idx = np.nonzero(dl.labels_train['mask'])[0]
    np.random.shuffle(train_idx)
    split = int(train_idx.shape[0]*val_ratio)
    val_idx = train_idx[:split]
    train_idx = train_idx[split:]
    train_idx = np.sort(train_idx)
    val_idx = np.sort(val_idx)
    test_idx = np.nonzero(dl.labels_test['mask'])[0]
    labels[train_idx] = dl.labels_train['data'][train_idx]
    labels[val_idx] = dl.labels_train['data'][val_idx]
    labels[test_idx] = dl.labels_test['data'][test_idx]
    # if prefix != 'IMDB':
    #     labels = labels.argmax(axis=1)
    train_val_test_idx = {}
    train_val_test_idx['train_idx'] = train_idx
    train_val_test_idx['val_idx'] = val_idx
    train_val_test_idx['test_idx'] = test_idx

    return features,\
        adjM, \
        labels,\
        train_val_test_idx,\
        dl


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


# compute the regression loss
def regression_loss(logits, labels):
    return F.mse_loss(logits, labels)
