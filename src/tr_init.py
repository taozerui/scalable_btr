import numpy as np
from tensorly.decomposition import tensor_ring


def convert_idx2tensor(x_val, idx, sz):
    tensor = np.zeros(sz)
    tensor[tuple(np.transpose(idx))] = x_val.reshape(-1)
    return tensor


def tr_als(x_val, idx, sz, rank):
    dim = len(sz)
    if isinstance(rank, int):
        rank = [rank] * dim

    # init
    cores = []


def tr_svd(x_val, idx, sz, rank):
    tensor = convert_idx2tensor(x_val, idx, sz)
    tr = tensor_ring(tensor, rank)
    return tr


if __name__ == '__main__':
    from scipy.io import loadmat

    data = loadmat('../data/alog/alog_fold1.mat')
    train_idx = data['train_idx'] - 1
    train_vals = data['train_vals']
    test_idx = data['test_idx'] - 1
    test_vals = data['test_vals']

    sz = [200, 100, 200]
    dim = 3

    tensor = convert_idx2tensor(train_vals, train_idx, sz)
    tr = tr_svd(train_vals, train_idx, sz, 10)
