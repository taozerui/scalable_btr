import pickle
import torch
import numpy as np
from pathlib import Path

from src.batch_btr import BatchGibbsBTR
from src.tensor import full_tr
from src.utils import gen_mask

torch.random.manual_seed('123')
np.random.seed(42)


def gen_cores(sz, rank, method='randn'):
    cores = []
    dim = len(sz)
    for i in range(dim):
        if method == 'randn':
            cores.append(torch.randn(sz[i], rank[i-1], rank[i]))
        elif method == 'rand':
            cores.append(torch.randn(sz[i], rank[i - 1], rank[i]))
    return cores


def generate_continuous_data(sz, rank, snr):
    cores = gen_cores(sz, rank)
    x = full_tr(cores)
    x = x / torch.sqrt(torch.var(x))
    sigma_x = torch.var(x)
    scale = sigma_x / (10 ** (snr / 10.))
    noise = torch.randn(sz) * torch.sqrt(scale)
    return x, x + noise, 1 / scale


if __name__ == '__main__':
    dim = 4
    sz = [10] * dim
    rank = [5] * dim
    device = torch.device('cuda:5')

    SNR = [10, 15, 20, 25, 30]
    MSR = [0.1, 0.3, 0.5, 0.7, 0.9]
    rank_est_error = {}
    data = {}
    for snr in SNR:
        for msr in MSR:
            x_low_rank, x_noise, tau = generate_continuous_data(sz, rank, snr)
            x_low_rank = x_low_rank.to(device)
            x_noise = x_noise.to(device)
            tau = tau.to(device)
            mask = gen_mask(x_noise, msr).to(device)
            x_obs = x_noise[mask == 1.0].to(device)
            idx = torch.tensor(np.argwhere(mask.cpu().numpy() == 1.0)).to(device)

            x_val = x_low_rank[mask == 0.0].to(device)
            val_idx = torch.tensor(np.argwhere(mask.cpu().numpy() == 0.0)).to(device)
            init_rank = [2] * dim
            trd = BatchGibbsBTR(
                dims=sz, init_rank=init_rank,
                a_delta=3.1, a_noise=1.0, b_noise=0.3,
            ).to(device)
            trd.fit(
                x_obs,
                idx,
                burn_in=1500,
                sample_num=100,
                init_method='rand',
                init_scale=0.2,
                tune_method='adap',  # 'truncate',
                truncate_tol=1e-2,
            )
            rank_hat = trd.get_rank()
            err = np.abs(np.array(rank_hat) - np.array(rank)).sum() / np.array(rank).sum()
            rank_est_error[f'snr-{snr:.2f}_mr-{msr:.2f}'] = err
            data[f'snr_{snr}_mr_{int(100*msr)}_noise'] = x_noise.cpu().numpy()
            data[f'snr_{snr}_mr_{int(100*msr)}_x'] = x_low_rank.cpu().numpy()
            data[f'snr_{snr}_mr_{int(100*msr)}_mask'] = mask.cpu().numpy()
    
    Path(f'./results/simulation').mkdir(parents=True, exist_ok=True)
    with open(f'./results/simulation/rank_error_init{init_rank[0]}.pkl', 'wb') as f:
        pickle.dump(rank_est_error, f)
