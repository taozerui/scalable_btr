import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch import nn
from tqdm import tqdm
from decimal import Decimal

from src.batch_btr import BayesTensorRing


class SparseBTR(BayesTensorRing):
    """ SparseBTR """
    def __init__(
            self,
            dims,
            init_rank,
            a_delta=2.1,
            a_noise=1e-6,
            b_noise=1e-6,
            data_type='continuous',
            dtype=torch.float32):
        super(SparseBTR, self).__init__(dims, init_rank, data_type, dtype)

        self._init_params()

        self.hyper_prior = {
            'a_delta': a_delta,
            'a_noise': a_noise,
            'b_noise': b_noise,
        }

        self.logger = {'loss': [],  'train_mse': [],
                       'val_mse': [], 'val_rmse': [], 'val_mae': [],
                       'val_auc': [], 'val_acc':[],
                       'val_ce': []}
        self.core_mean = None

    def _init_params(self):
        # lambda
        lambda_list = []
        for d in range(self.order):
            lambda_list.append(
                nn.Parameter(torch.ones(self.init_rank[d], dtype=self.dtype), requires_grad=True)
            )
        self.lambda_list = nn.ParameterList(lambda_list)

        delta = []
        for d in range(self.order):
            delta.append(
                nn.Parameter(torch.ones(self.init_rank[d], dtype=self.dtype), requires_grad=False)
            )
        self.delta = nn.ParameterList(delta)

        # noise precision
        # self.log_noise_precision = nn.Parameter(torch.zeros(1, dtype=self.dtype), requires_grad=True)
        self.log_noise_precision = nn.Parameter(torch.tensor([0.0], dtype=self.dtype), requires_grad=False)

    def fit(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        max_epoch: int = 1000,
        batch_size: int = 128,
        lr: float = 1e-3,
        lr_aneal: float = 1.0,
        init_method: str = 'rand',
        init_scale: float = 0.2,
        tol: float = 1e-3,
        val_x=None,
        val_mask=None,
        init_core=True,
    ):
        if init_core:
            self.init_cores_(method=init_method, scale=init_scale, vals=x, idx=mask)
        for p in self.cores:
            p.requires_grad = True

        num_obs = x.shape[0]

        shuffle = torch.randperm(mask.shape[0])
        x = x[shuffle]
        mask = mask[shuffle]
        x_batch = torch.split(x, split_size_or_sections=batch_size)
        mask_batch = torch.split(mask, split_size_or_sections=batch_size)
        batch_num = len(x_batch)

        # optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        optimizer = torch.optim.Adam(
            # nn.ParameterList(list(self.cores) + list(self.lambda_list) + [self.log_noise_precision]),
            nn.ParameterList(list(self.cores) + list(self.lambda_list)),
            lr=lr
        )
        bar = tqdm(range(max_epoch), desc='Train')
        for step in bar:

            mse = 0.
            for batch in range(batch_num):
                rank = self.get_rank()

                # e-step
                # update PG variable
                x_hat = self(mask_batch[batch])
                if self.data_type == 'binary':
                    omega = 0.5 / x_hat * torch.tanh(x_hat * 0.5)
                else:
                    omega = None

                # update delta
                for d in range(self.order):
                    for r in range(rank[d]):
                        a_delta = self.hyper_prior['a_delta'] + 0.5 * (rank[d] - (r + 1) + 1)
                        b_delta = 0.
                        for h in range(r, rank[d]):
                            b_delta += self.lambda_list[d][h] ** 2 * torch.prod(self.delta[d][:h]) / self.delta[d][r]
                        b_delta = 1. + 0.5 * b_delta
                        self.delta[d][r].data = a_delta / b_delta

                # m-step
                if self.data_type == 'continuous':
                    loss = self.free_energy(x_batch[batch], x_hat, num_obs)
                else:
                    loss = self.free_energy_pg(x_batch[batch], x_hat, num_obs)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if self.data_type == 'continuous':
                    mse += torch.sum((x_hat - x_batch[batch]) ** 2).cpu().detach().item()
                else:
                    x_hat_prob = 1 / (1 + torch.exp(- x_hat))
                    mse += F.binary_cross_entropy(input=x_hat_prob,
                                                  target=x_batch[batch]).sum().cpu().detach().item()

                self.logger['loss'].append(loss.item())
                bar.set_postfix_str(f'loss={Decimal(loss.item()):.5E}.')

            mse /= num_obs
            self.logger['train_mse'].append(mse)
            if val_x is not None:
                val_x_hat = self(val_mask)
                if self.data_type == 'continuous':
                    val_mse = torch.mean((val_x_hat - val_x) ** 2).item()
                    val_rmse = np.sqrt(val_mse)
                    val_mae = torch.mean(torch.abs(val_x_hat - val_x)).item()
                    self.logger['val_mse'].append(val_mse)
                    self.logger['val_mae'].append(val_mae)
                    self.logger['val_rmse'].append(val_rmse)
                    print(f'Val: MSE - {val_mse:.5f} | RMSE - {val_rmse:.5f} | MAE - {val_mae:.5f}.')
                else:
                    val_prob_hat = 1 / (1 + torch.exp(- val_x_hat))
                    val_auc = roc_auc_score(y_true=val_x.cpu().detach().numpy(),
                                            y_score=val_prob_hat.cpu().detach().numpy())
                    val_label_hat = val_prob_hat.clone()
                    val_label_hat[val_label_hat > 0.5] = 1.0
                    val_label_hat[val_label_hat <= 0.5] = 0.0
                    val_acc = torch.sum(val_x.cpu().detach() == val_label_hat.cpu().detach()) / val_x.shape[0]
                    val_mse = torch.mean((val_x - val_prob_hat) ** 2).item()
                    val_mae = torch.mean(torch.abs(val_x - val_prob_hat)).item()
                    val_rmse = np.sqrt(val_mse)
                    val_ce = - torch.mean(
                        val_x * torch.log(val_prob_hat + 1e-10) + 
                        (1 - val_x) * torch.log(1 - val_prob_hat + 1e-10)
                    ).item()
                    self.logger['val_auc'].append(val_auc)
                    self.logger['val_acc'].append(val_acc)
                    self.logger['val_mse'].append(val_mse)
                    self.logger['val_mae'].append(val_mae)
                    self.logger['val_rmse'].append(val_rmse)
                    self.logger['val_ce'].append(val_ce)
                    print(f'Val: AUC - {val_auc:.5f} | ACC - {val_acc:.5f} | BCE - {val_ce:.5f} | RMSE - {val_rmse:.5f} | MAE - {val_mae:.5f}.')

            # anneal learning rate
            if step % 10 == 0:
                optimizer.param_groups[0]['lr'] *= lr_aneal

    def free_energy(self, x, x_hat, num_obs):
        batch_size = x_hat.shape[0]
        rank = self.get_rank()
        noise_precision = torch.exp(self.log_noise_precision)
        loss = 0.5 * noise_precision * num_obs / batch_size * ((x - x_hat) ** 2).sum()
        for d, core in enumerate(self.cores):
            # core prior
            loss = loss + 0.5 * torch.norm(core, p=2) ** 2
            # lambda prior
            for h in range(rank[d]):
                tau = torch.prod(self.delta[d][:h])
                loss = loss + 0.5 * tau * self.lambda_list[d][h] ** 2

        loss = loss - (self.hyper_prior['a_noise'] - 1) * self.log_noise_precision \
            + self.hyper_prior['b_noise'] * noise_precision
        return loss

    def free_energy_pg(self, x, x_hat, num_obs):
        batch_size = x.shape[0]
        rank = self.get_rank()
        # loss = (x * torch.log(1. + torch.exp(- x_hat)) +
        #         (1 - x) * torch.log(1. + torch.exp(x_hat))).sum() / batch_size * num_obs
        omega = (0.5 / x_hat * torch.tanh(x_hat * 0.5)).detach()
        omega[torch.isnan(omega)] = 0.0
        kappa = x - 0.5
        loss = (0.5 * omega * x_hat ** 2 - kappa * x_hat).sum() / batch_size * num_obs
        for d, core in enumerate(self.cores):
            # core prior
            loss = loss + 0.5 * torch.norm(core, p=2) ** 2
            # lambda prior
            for h in range(rank[d]):
                tau = torch.prod(self.delta[d][:h])
                loss = loss + 0.5 * tau * self.lambda_list[d][h] ** 2
        return loss
