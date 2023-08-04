import torch
import numpy as np
from cprint import cprint

from src.batch_btr import BatchGibbsBTR
from src.sparse_btr import SparseBTR

torch.random.manual_seed('123')
np.random.seed(42)

# settings
sz = [203, 203, 200]
dim = 3
device = 'cuda:4'
dtype = torch.float32
device = torch.device(device)
init_rank = [5] * dim

# data
data = np.load('./data/enron/fold-0.npz')
train = data['train']
test = data['test']
train_val = train[:, -1].astype('float64')
train_idx = train[:, :-1].astype('int64')
test_val = test[:, -1].astype('float64')
test_idx = test[:, :-1].astype('int64')

train_val = torch.tensor(train_val, dtype=dtype).to(device)
train_idx = torch.tensor(train_idx, dtype=torch.int64).to(device)
test_val = torch.tensor(test_val, dtype=dtype).to(device)
test_idx = torch.tensor(test_idx, dtype=torch.int64).to(device)

# run
cprint('Running Online EM algorithm...', c='r')
trd_gibbs = BatchGibbsBTR(
    dims=sz, init_rank=init_rank,
    a_delta=2.0, a_noise=1.0, b_noise=0.3, data_type='binary', dtype=dtype
).to(device)
trd_gibbs.fit(
    x=train_val,
    idx=train_idx,
    burn_in=500,
    sample_num=100,
    tune_method='none',
    truncate_tol=1e-3,
    init_method='randn',
    init_scale=0.5,
    val_x=test_val,
    val_idx=test_idx
)

cprint('Running Online EM algorithm...', c='r')
trd_em = SparseBTR(
    dims=sz, init_rank=init_rank, dtype=dtype, data_type='binary'
).to(device)
trd_em.fit(
    train_val, train_idx,
    max_epoch=500, batch_size=512, lr=1e-2,
    init_method='rand', init_scale=0.5,
    val_x=test_val, val_mask=test_idx,
    lr_aneal=1.,
)
