import torch
import numpy as np
from time import time
from torch import nn
import torch.nn.functional as F
from torch.distributions import Gamma, Normal
from polyagamma import random_polyagamma
from sklearn.metrics import roc_auc_score
from typing import List, Optional, Union, Tuple
from decimal import Decimal

from src.tensor import get_left_chain_idx, full_tr_idx, full_tr
from src.tr_init import tr_svd
from src.tr import TrALS


class BayesTensorRing(nn.Module):
    """ BayesTensorRing """
    def __init__(
        self,
        dims: List[int],
        init_rank: Union[int, List[int]],
        data_type: str = 'continuous',  # continuous or binary
        dtype: torch.dtype = torch.float64,
    ):
        super(BayesTensorRing, self).__init__()
        self.dims = dims
        self.order = len(dims)
        if isinstance(init_rank, int):
            init_rank = [init_rank] * len(dims)
        assert len(init_rank) == len(dims), "Length of rank not compatible."
        self.init_rank = init_rank
        self.data_type = data_type
        self.dtype = dtype

        self._setup_cores()

        self.cores_mean = None
        self.lambda_mean = None
        self.have_post_samples = False
        self.logger = {'MSE': [], 'noise': [], 'AUC': [], 'Val_Loss': [],
                       'val_rmse': [], 'val_mse': [], 'val_mae':[],
                       'val_auc': [], 'val_acc':[], 'val_ce': []}

    def get_rank(self):
        rank = []
        for core in self.cores:
            rank.append(core.shape[-1])

        return rank

    def _setup_cores(self):
        cores = []
        for d in range(self.order):
            core = torch.empty(self.dims[d], self.init_rank[d-1], self.init_rank[d], dtype=self.dtype)
            cores.append(nn.Parameter(core, requires_grad=False))

        self.cores = nn.ParameterList(cores)

    def init_cores_(self, method: str = 'rand', scale: float = 0.2, vals=None, idx=None):
        if method == 'svd':
            tr = tr_svd(x_val=vals.cpu(), idx=idx.cpu(), sz=self.dims, rank=self.init_rank + [self.init_rank[0]])
        elif method == 'als':
            print('ALS init...')
            tr = TrALS(dims=self.dims, init_rank=self.get_rank(), dtype=self.dtype).fit(
                x=vals.cpu(), idx=idx.cpu(), max_epoch=10, init_scale=scale
            )
        else:
            tr = None
        for d, core in enumerate(self.cores):
            if method == 'rand':
                torch.nn.init.uniform_(core, 0.0, scale)
            elif method == 'randn':
                torch.nn.init.normal_(core, 0.0, scale)
            elif method == 'xavier':
                torch.nn.init.xavier_normal_(core)
            elif method == 'svd':
                assert tr is not None
                core.data = torch.tensor(tr.factors[d].copy().transpose([1, 0, 2]), dtype=self.dtype)
            elif method == 'als':
                assert tr is not None
                core.data = torch.tensor(tr.cores[d].clone().to(core.device), dtype=self.dtype)
            else:
                raise NotImplementedError

    def forward(self, index: Optional[torch.Tensor] = None):
        assert self.cores is not None, "Fit the model first!"

        if not self.have_post_samples:
            cores = self.cores
            lambda_list = self.lambda_list
        else:
            cores = self.cores_mean
            lambda_list = self.lambda_mean

        if index is None:
            return full_tr(cores, lambda_list)
        else:
            return full_tr_idx(cores, index, lambda_list)

    def fit(self, *args, **kwargs):
        raise NotImplementedError

    def validate(self, val_x, val_idx):
        val_hat = self(val_idx)
        if self.data_type == 'continuous':
            val_mse = torch.mean((val_x - val_hat) ** 2).item()
            val_mae = torch.mean(torch.abs(val_x - val_hat)).item()
            val_rmse = np.sqrt(val_mse)
            self.logger['val_mse'].append(val_mse)
            self.logger['val_mae'].append(val_mae)
            self.logger['val_rmse'].append(val_rmse)
            print(f'Val: MSE - {val_mse:.5f} | RMSE - {val_rmse:.5f} | MAE - {val_mae:.5f}.\n')
        else:
            val_x_hat = 1 / (1 + torch.exp(- val_hat))
            val_auc = roc_auc_score(
                y_true=val_x.cpu().numpy(), y_score=val_x_hat.cpu().numpy()
            )
            val_label_hat = val_x_hat.clone()
            val_label_hat[val_label_hat > 0.5] = 1.0
            val_label_hat[val_label_hat <= 0.5] = 0.0
            val_acc = torch.sum(val_x.cpu().detach() == val_label_hat.cpu().detach()) / val_x.shape[0]
            val_mse = torch.mean((val_x - val_x_hat) ** 2).item()
            val_mae = torch.mean(torch.abs(val_x - val_x_hat)).item()
            val_rmse = np.sqrt(val_mse)
            val_ce = - torch.mean(
                val_x * torch.log(val_x_hat + 1e-10) + (1 - val_x) * torch.log(1 - val_x_hat + 1e-10)
            ).item()
            self.logger['val_auc'].append(val_auc)
            self.logger['val_acc'].append(val_acc)
            self.logger['val_mse'].append(val_mse)
            self.logger['val_mae'].append(val_mae)
            self.logger['val_rmse'].append(val_rmse)
            self.logger['val_ce'].append(val_ce)
            print(f'Val: AUC - {val_auc:.5f} | ACC - {val_acc:.5f} | BCE - {val_ce:.5f} | RMSE - {val_rmse:.5f} | MAE - {val_mae:.5f}.')

    def _adap_tune_rank(self, step, tol, b0=1.0, b1=0.005):
        prob = 1. / np.exp(b0 + b1 * step)
        device = self.cores[0].device

        for d in range(self.order):
            if d == self.order - 1:
                dn = 0
            else:
                dn = d + 1
            weight = torch.abs(self.lambda_list[d])
            weight = weight / torch.max(weight)
            preserve_idx = weight > tol

            if preserve_idx.all():  # if all the cores are large, we add new components
                if np.random.rand() < prob:  # add component with prob
                    self.lambda_list[d].data = torch.cat(
                        [self.lambda_list[d].data, torch.randn(1, dtype=self.dtype, device=device)]
                    )
                    self.delta[d].data = torch.cat(
                        [self.delta[d].data, torch.ones(1, dtype=self.dtype, device=device)]
                    )
                    self.cores[d].data = torch.cat(
                        [self.cores[d].data,
                         torch.randn(self.cores[d].shape[0], self.cores[d].shape[1], 1,
                                     dtype=self.dtype, device=device) * 0.2],
                        dim=-1
                    )
                    self.cores[dn].data = torch.cat(
                        [self.cores[dn].data,
                         torch.randn(self.cores[dn].shape[0], 1, self.cores[dn].shape[2],
                                     dtype=self.dtype, device=device) * 0.2],
                        dim=-2
                    )
            else:  # truncate
                self.lambda_list[d].data = self.lambda_list[d].data[preserve_idx]
                self.cores[d].data = self.cores[d].data[:, :, preserve_idx]
                self.cores[dn].data = self.cores[dn].data[:, preserve_idx, :]
                self.delta[d].data = self.delta[d].data[preserve_idx]

    def _trun_tune_rank(self, tol):
        for d in range(self.order):
            preserve_idx = torch.abs(self.lambda_list[d]) > tol
            if d == self.order - 1:
                dn = 0
            else:
                dn = d + 1

            self.lambda_list[d].data = self.lambda_list[d].data[preserve_idx]
            self.cores[d].data = self.cores[d].data[:, :, preserve_idx]
            self.cores[dn].data = self.cores[dn].data[:, preserve_idx, :]
            self.delta[d].data = self.delta[d].data[preserve_idx]

    def tune_rank(self, method, step, tol):
        if (step + 1) % 10 == 0:
            if method == 'adap':
                self._adap_tune_rank(step + 1, tol)
            elif method == 'truncate':
                self._trun_tune_rank(tol)
            elif method == 'none':
                pass
            else:
                raise NotImplementedError
        else:
            pass

    def _init_priors(self):
        # lambda
        lambda_list = []
        for d in range(self.order):
            lambda_list.append(
                nn.Parameter(torch.ones(self.init_rank[d], dtype=self.dtype), requires_grad=False)
            )
        self.lambda_list = nn.ParameterList(lambda_list)

        # delta
        delta = []
        for d in range(self.order):
            delta.append(
                nn.Parameter(torch.ones(self.init_rank[d], dtype=self.dtype), requires_grad=False)
            )
        self.delta = nn.ParameterList(delta)

        # psi
        self.psi = nn.Parameter(torch.ones([], dtype=self.dtype), requires_grad=False)

        # noise precision
        if self.data_type == 'continuous':
            self.noise_precision = nn.Parameter(torch.ones(1, dtype=self.dtype), requires_grad=False)
        elif self.data_type == 'binary':
            self.noise_precision = None
        else:
            raise NotImplementedError

    def collect_sample_mean(self, step, jump):
        if step % jump == 0:
            for d in range(self.order):
                self.cores_mean[d] += self.cores[d]
                self.lambda_mean[d] += self.lambda_list[d]
            self.sample_count += 1

    def sample_delta(self, is_sample=True):
        rank = self.get_rank()
        for d in range(self.order):
            for r in range(rank[d]):
                a_delta = self.hyper_prior['a_delta'] + 0.5 * (rank[d] - (r + 1) + 1)
                b_delta = 0.
                for h in range(r, rank[d]):
                    b_delta += self.lambda_list[d][h] ** 2 * \
                               torch.prod(self.delta[d][:r]) * torch.prod(self.delta[d][r+1:h])
                b_delta = 1. + 0.5 * b_delta
                if is_sample:
                    self.delta[d].data[r] = Gamma(concentration=a_delta, rate=b_delta).sample()
                else:
                    self.delta[d].data[r] = a_delta / b_delta

    def sample_lambda(self, x, idx, omega, is_sample=True):
        rank = self.get_rank()
        left_chain = get_left_chain_idx(idx, self.cores, self.lambda_list)
        right_chain = None
        right_chain_update = []
        for d in range(self.order):
            idx_d = idx[:, d]
            if d == 0:
                sub_chain = left_chain[d]
            elif d == self.order - 1:
                sub_chain = right_chain
            else:
                sub_chain = torch.einsum('imn, ink-> imk', left_chain[d], right_chain)

            sub_chain = torch.einsum('imn, ink-> imk', sub_chain, self.cores[d][idx_d, :, :])
            sub_chain = torch.diagonal(sub_chain, dim1=1, dim2=2)

            for r in range(rank[d]):
                a_r = sub_chain[:, r]
                self.lambda_list[d].data[r] = 0.
                b_r = torch.einsum('in, n-> i', sub_chain, self.lambda_list[d])

                if self.data_type == 'binary':
                    lambda_var_inv = torch.prod(self.delta[d][:r]) + (a_r ** 2 * omega).sum()
                    lambda_mean = (1. / lambda_var_inv) * (a_r * (x - 0.5 - omega * b_r)).sum()
                else:
                    lambda_var_inv = torch.prod(self.delta[d][:r]) + self.noise_precision * (a_r ** 2).sum()
                    lambda_mean = (1. / lambda_var_inv) * self.noise_precision * (a_r * (x - b_r)).sum()

                if torch.isnan(lambda_mean).any():
                    import ipdb; ipdb.set_trace()

                if is_sample:
                    self.lambda_list[d].data[r] = torch.randn_like(lambda_mean) / torch.sqrt(lambda_var_inv) + lambda_mean
                else:
                    self.lambda_list[d].data[r] = lambda_mean

            # update subchain
            if d != self.order - 1:
                if right_chain is None:
                    right_chain = torch.einsum('imn, n-> imn', self.cores[d][idx_d, :, :], self.lambda_list[d])
                else:
                    right_chain = torch.einsum('imn, ink, k-> imk',
                                               right_chain, self.cores[d][idx_d, :, :], self.lambda_list[d])
                right_chain_update.append(right_chain)  # Q_1, Q_{1, 2}, ..., Q_{1, ..., D-1}

        return right_chain_update

    def sample_core_tensor(self, x, idx, omega, right_chain_update, is_sample=True):
        rank = self.get_rank()
        # we update from D to 1, so that we can use the stored right chains
        left_chain = None
        for d in range(self.order - 1, -1, -1):
            idx_d = idx[:, d]
            if d == 0:
                sub_chain = left_chain
            elif d == self.order - 1:
                sub_chain = right_chain_update[d - 1]
            else:
                sub_chain = torch.einsum('imn, ink-> imk', left_chain, right_chain_update[d - 1])

            sub_chain = torch.einsum('n, ink-> ink', self.lambda_list[d], sub_chain)

            # tau_r = torch.zeros(self.dims[d], dtype=self.dtype, device=self.cores[0].device)
            # alpha_r = torch.zeros_like(tau_r)
            for r1 in range(rank[d - 1]):
                for r2 in range(rank[d]):
                    c_rr = sub_chain[:, r2, r1]

                    # foo = self.cores[d].clone()
                    # self.cores[d].data[:, r1, r2] = 0.
                    # d_rr = torch.einsum('imn, inm-> i', sub_chain, self.cores[d][idx_d, :, :])
                    d_rr = torch.einsum('imn, inm-> i', sub_chain, self.cores[d][idx_d, :, :])
                    d_rr -= c_rr * self.cores[d][idx_d, r1, r2]

                    for i_d in range(self.dims[d]):
                        idx_cur = idx_d == i_d

                        c_rr_cur = c_rr[idx_cur]
                        d_rr_cur = d_rr[idx_cur]

                        if self.data_type == 'binary':
                            tau_r = 1 + (omega[idx_cur] * c_rr_cur ** 2).sum()
                            alpha_r = 1. / tau_r * (
                                    c_rr_cur * (x[idx_cur] - 0.5 - omega[idx_cur] * d_rr_cur)
                            ).sum()
                        else:
                            tau_r = 1 + self.noise_precision * (c_rr_cur ** 2).sum()
                            alpha_r = 1. / tau_r * self.noise_precision * (
                                    c_rr_cur * (x[idx_cur] - d_rr_cur)
                            ).sum()
                        if is_sample:
                            self.cores[d].data[i_d, r1, r2] = torch.randn_like(alpha_r) / torch.sqrt(tau_r) + alpha_r
                        else:
                            self.cores[d].data[i_d, r1, r2] = alpha_r

            if left_chain is None:
                left_chain = torch.einsum('imn, n-> imn', self.cores[-1][idx_d, :, :], self.lambda_list[-1])
            else:
                left_chain = torch.einsum('imn, n, ink-> imk',
                                          self.cores[d][idx_d, :, :], self.lambda_list[d], left_chain)

        return left_chain


class BatchGibbsBTR(BayesTensorRing):
    def __init__(
        self,
        dims: List[int],
        init_rank: Union[int, List[int]],
        data_type: str = 'continuous',  # continuous or binary
        dtype: torch.dtype = torch.float64,
        a_delta: float = 2.1,
        a_noise: float = 1.0,
        b_noise: float = 0.3,
        sigma_core: float = 1.0,
    ):
        super(BatchGibbsBTR, self).__init__(dims, init_rank, data_type, dtype)

        self._init_priors()

        self.sample_count = 0

        self.hyper_prior = {
            'a_delta': a_delta,
            'a_noise': a_noise,
            'b_noise': b_noise,
            'sigma_core': sigma_core,
            'a_psi': 1e-3,
            'b_psi': 1e-3
        }

    def fit(
        self,
        x: torch.Tensor,
        idx: torch.Tensor,
        burn_in: int = 1000,
        sample_num: int = 500,
        jump: int = 5,
        tune_method: str = 'adap',  # 'adap' or 'truncate'
        init_method: str = 'rand',
        init_scale: float = 0.2,
        truncate_tol: float = 1e-3,
        val_x: Optional[torch.Tensor] = None,
        val_idx: Optional[torch.Tensor] = None
    ):
        self.init_cores_(method=init_method, scale=init_scale, vals=x.cpu(), idx=idx.cpu())

        obs_num = idx.shape[0]
        assert x.shape[0] == obs_num

        x_hat = None
        for step in range(burn_in + sample_num):
            tic = time()
            rank = self.get_rank()

            # Polya Gamma augmentation
            if self.data_type == 'binary':
                if x_hat is None:
                    x_hat = self(idx)
                omega = torch.tensor(random_polyagamma(1, x_hat.cpu().numpy()),
                                     dtype=self.dtype, device=x_hat.device)
            else:
                omega = None

            # update delta
            self.sample_delta()

            # update lambda
            right_chain_update = self.sample_lambda(x, idx, omega)

            # update core tensors
            left_chain = self.sample_core_tensor(x, idx, omega, right_chain_update)

            # update noise
            # x_hat = self(idx)
            x_hat = torch.einsum('inn-> i', left_chain)
            if self.data_type == 'continuous':
                err = ((x_hat - x) ** 2).sum()
                a_epsilon = self.hyper_prior['a_noise'] + 0.5 * obs_num
                b_epsilon = self.hyper_prior['b_noise'] + 0.5 * err
                self.noise_precision.data = Gamma(concentration=a_epsilon, rate=b_epsilon).sample()
            else:
                prob_hat = 1 / (1 + torch.exp(- x_hat))
                # err = roc_auc_score(x.cpu().numpy(), prob_hat.cpu().numpy())
                err = F.binary_cross_entropy(input=prob_hat, target=x).cpu().item()

            # tune rank and collect samples
            if step < burn_in:
                self.tune_rank(tune_method, step, truncate_tol)
            elif step == burn_in:  # init
                self.cores_mean = []
                self.lambda_mean = []
                for d in range(self.order):
                    self.cores_mean.append(torch.zeros_like(self.cores[d]))
                    self.lambda_mean.append(torch.zeros_like(self.lambda_list[d]))
            else:
                self.collect_sample_mean(step - burn_in, jump)

            # log result
            toc = time()
            if self.data_type == 'continuous':
                self.logger['MSE'].append(err.item() / obs_num)
                self.logger['noise'].append(self.noise_precision.data.item())
                print(f'Iter. {step} - Elapsed time {toc - tic:.2f}s...')
                print(f'Rank {rank}.')
                print(f'MSE is {Decimal(err.item() / obs_num):.5E}.')
            else:
                self.logger['AUC'].append(err)
                print(f'Iter. {step} - Elapsed time {toc - tic:.2f}s...')
                print(f'Rank {rank}.')
                print(f'BCE is {Decimal(err):.5E}.')

            # validate
            if val_x is not None and val_idx is not None:
                self.validate(val_x, val_idx)

        for d in range(self.order):
            self.cores_mean[d] /= self.sample_count
            self.lambda_mean[d] /= self.sample_count
        self.have_post_samples = True


class BatchEMBTR(BayesTensorRing):
    def __init__(
            self,
            dims: List[int],
            init_rank: Union[int, List[int]],
            data_type: str = 'continuous',  # continuous or binary
            dtype: torch.dtype = torch.float64,
            a_delta: float = 2.1,
            a_noise: float = 1.0,
            b_noise: float = 0.3,
            sigma_core: float = 1.0,
    ):
        super(BatchEMBTR, self).__init__(dims, init_rank, data_type, dtype)

        self._init_priors()

        self.hyper_prior = {
            'a_delta': a_delta,
            'a_noise': a_noise,
            'b_noise': b_noise,
            'sigma_core': sigma_core,
            'a_psi': 1e-3,
            'b_psi': 1e-3
        }

    def fit(
            self,
            x: torch.Tensor,
            idx: torch.Tensor,
            max_iter: int = 200,
            tune_method: str = 'adap',  # 'adap' or 'truncate'
            init_method: str = 'rand',
            init_scale: float = 0.2,
            truncate_tol: float = 1e-3,
            val_x: Optional[torch.Tensor] = None,
            val_idx: Optional[torch.Tensor] = None
    ):
        self.init_cores_(method=init_method, scale=init_scale, vals=x, idx=idx)

        obs_num = idx.shape[0]
        assert x.shape[0] == obs_num

        x_hat = None
        for step in range(max_iter):
            tic = time()
            rank = self.get_rank()

            # Polya Gamma augmentation
            if self.data_type == 'binary':
                if x_hat is None:
                    x_hat = self(idx)
                omega = 0.5 / x_hat * torch.tanh(x_hat * 0.5)
                # omega[omega == torch.nan] = 0.0
                omega[torch.isnan(omega)] = 0.0
                if torch.isnan(omega).any():
                    import ipdb; ipdb.set_trace()
            else:
                omega = None

            # update delta
            self.sample_delta(is_sample=False)

            # update lambda
            right_chain_update = self.sample_lambda(x, idx, omega, is_sample=False)

            # update core tensors
            left_chain = self.sample_core_tensor(x, idx, omega, right_chain_update, is_sample=False)

            # update noise
            x_hat = torch.einsum('inn-> i', left_chain)
            if self.data_type == 'continuous':
                err = ((x_hat - x) ** 2).sum()
                a_epsilon = self.hyper_prior['a_noise'] + 0.5 * obs_num
                b_epsilon = self.hyper_prior['b_noise'] + 0.5 * err
                self.noise_precision.data = a_epsilon / b_epsilon
            else:
                prob_hat = 1 / (1 + torch.exp(- x_hat))
                try:
                    err = F.binary_cross_entropy(input=prob_hat, target=x).cpu().item()
                except:
                    import ipdb; ipdb.set_trace()

            # tune rank and collect samples
            if step < max_iter - 50:
                self.tune_rank(tune_method, step, truncate_tol)

            # log result
            toc = time()
            if self.data_type == 'continuous':
                self.logger['MSE'].append(err.item() / obs_num)
                self.logger['noise'].append(self.noise_precision.data.item())
                print(f'Iter. {step} - Elapsed time {toc - tic:.2f}s...')
                print(f'Rank {rank}.')
                print(f'MSE is {Decimal(err.item() / obs_num):.5E}.\n')
            else:
                self.logger['AUC'].append(err)
                print(f'Iter. {step} - Elapsed time {toc - tic:.2f}s...')
                print(f'Rank {rank}.')
                print(f'AUC is {Decimal(err):.5E}.')

            # validate
            if val_x is not None and val_idx is not None:
                self.validate(val_x, val_idx)
