import torch
import numpy as np
from scipy.io import loadmat
from cprint import cprint

from src.batch_btr import BatchGibbsBTR
from src.sparse_btr import SparseBTR
from src.utils import gen_mask

torch.random.manual_seed('123')
np.random.seed(42)

# settings
sz = [17, 125, 156]
dim = len(sz)
msr = 0.9
device = 'cuda:4'
dtype = torch.float32
device = torch.device(device)
init_rank = [5] * dim

# data
data = loadmat('./data/USHCN.mat')
X = []
for i in range(17):
    X.append(data['data_series'][i][0])
X = np.array(X)
X = torch.tensor(X, dtype=dtype).to(device).view(sz)

mask = gen_mask(X, msr)
X_obs = X[mask == 1.0]
idx = torch.tensor(np.argwhere(mask.cpu().numpy() == 1.0)).to(device)
X_val = X[mask == 0.0]
idx_val = torch.tensor(np.argwhere(mask.cpu().numpy() == 0.0)).to(device)

# run
cprint('Running Gibbs Sampler...', c='r')
trd_gibbs = BatchGibbsBTR(
    dims=sz, init_rank=init_rank,
    a_delta=2.0, a_noise=1.0, b_noise=0.3
).to(device)
trd_gibbs.fit(
    x=X_obs,
    idx=idx,
    burn_in=200,
    sample_num=50,
    tune_method='adap',  # 'truncate',
    truncate_tol=1e-2,
    init_method='rand',
    init_scale=0.5,
    val_x=X_val,
    val_idx=idx_val
)

cprint('Running Online EM algorithm...', c='r')
trd_em = SparseBTR(dims=sz, init_rank=init_rank, dtype=dtype).to(device)
trd_em.fit(
    X_obs, idx,
    max_epoch=200,
    batch_size=512,
    lr=3e-3,
    init_method='rand',
    init_scale=0.2,
    val_x=X_val,
    val_mask=idx_val
)