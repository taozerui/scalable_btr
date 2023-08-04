import torch
from torch import nn
from tqdm import tqdm
from typing import List, Union
from decimal import Decimal
from typing import Optional

from src.tensor import full_tr, full_tr_idx, get_left_chain_idx
from src.utils import get_batch_data


class TensorRing(nn.Module):
    """ TensorRing """
    def __init__(self, dims, init_rank, dtype):
        super(TensorRing, self).__init__()
        self.dims = dims
        self.order = len(dims)
        if isinstance(init_rank, int):
            rank = [init_rank] * self.order
        else:
            assert len(init_rank) == len(dims)
            rank = init_rank
        self.rank = rank
        self.dtype = dtype

        cores = []
        for d in range(self.order):
            cores.append(nn.Parameter(
                torch.empty(dims[d], rank[d-1], rank[d]), requires_grad=False
            ))
        self.cores = nn.ParameterList(cores)

    def init_cores_(self, method: str = 'rand', scale: float = 0.2):
        for d, core in enumerate(self.cores):
            if method == 'rand':
                torch.nn.init.uniform_(core, 0.0, scale)
            elif method == 'randn':
                torch.nn.init.normal_(core, 0.0, scale)
            elif method == 'xavier':
                torch.nn.init.xavier_normal_(core)
            else:
                raise NotImplementedError

    def forward(self, index: Optional[torch.Tensor] = None):

        if index is None:
            return full_tr(self.cores)
        else:
            return full_tr_idx(self.cores, index)

    def fit(self, *args, **kwargs):
        raise NotImplementedError


class TrGrad(TensorRing):
    """ TRD """
    def __init__(
        self,
        dims: List[int],
        init_rank: Union[int, List[int]],
        dtype: torch.dtype = torch.float64,
    ):
        super(TrGrad, self).__init__(dims, init_rank, dtype)
        self.logger = {'loss': []}

    def fit(
        self,
        x: torch.Tensor,
        idx: torch.Tensor,
        max_epoch: int = 1000,
        batch_size: int = 128,
        lr: float = 1e-3,
        init_method: str = 'rand',
        init_scale: float = 0.2,
    ):
        self.init_cores_(method=init_method, scale=init_scale)
        for p in self.cores:
            p.requires_grad = True

        x_batch, mask_batch = get_batch_data(x, idx, batch_size)
        batch_num = len(x_batch)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        bar = tqdm(range(max_epoch), desc='Train')
        for _ in bar:

            for batch in range(batch_num):

                x_hat = self(mask_batch[batch])
                loss = torch.mean((x_batch[batch] - x_hat) ** 2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                self.logger['loss'].append(loss.item())
                bar.set_postfix_str(f'loss={Decimal(loss.item()):.5E}.')

        for p in self.cores:
            p.requires_grad = False


class TrALS(TensorRing):
    """ TRD """
    def __init__(
        self,
        dims: List[int],
        init_rank: Union[int, List[int]],
        dtype: torch.dtype = torch.float64,
    ):
        super(TrALS, self).__init__(dims, init_rank, dtype)
        self.logger = {'loss': []}

    def fit(
        self,
        x: torch.Tensor,
        idx: torch.Tensor,
        max_epoch: int = 100,
        init_method: str = 'rand',
        init_scale: float = 0.2,
    ):
        self.init_cores_(method=init_method, scale=init_scale)
        bar = tqdm(range(max_epoch), desc='Train')
        for _ in bar:

            left_chains = get_left_chain_idx(idx, self.cores)
            right_chain = None
            # sweep from left to right
            for d in range(self.order):
                idx_d = idx[:, d]
                r1 = self.rank[d - 1]
                r2 = self.rank[d]

                if d == 0:
                    sub_chain = left_chains[d]
                elif d == self.order - 1:
                    sub_chain = right_chain
                else:
                    sub_chain = torch.einsum('imn, ink-> imk', left_chains[d], right_chain)

                sub_chain = sub_chain.permute([0, 2, 1]).flatten(-2)

                for i_d in range(self.dims[d]):
                    idx_i = idx_d == i_d

                    n = torch.sum(idx_i)

                    if n == 0:
                        pass
                    else:
                        sub_chain_id = sub_chain[idx_i, :]
                        xi = x[idx_i]
                        try:
                            # new_core = torch.matmul(
                            #     torch.linalg.inv(torch.einsum('im, in-> mn', sub_chain_id, sub_chain_id) / n),
                            #     torch.einsum('im, i-> m', sub_chain_id, xi) / n
                            # ).view(r1, r2)
                            # self.cores[d].data[i_d, :, :] = new_core
                            foo = torch.linalg.lstsq(sub_chain_id, xi.view(-1, 1)).solution  # driver='gelsd'
                            self.cores[d].data[i_d, :, :] = foo.view(r1, r2)
                        except RuntimeError:
                            pass

                if d == 0:
                    right_chain = self.cores[d][idx_d, :, :]
                else:
                    right_chain = torch.einsum('imn, ink-> imk', right_chain, self.cores[d][idx_d, :, :])

            x_hat = torch.einsum('inn-> i', right_chain)
            mse = torch.sum((x - x_hat) ** 2) / torch.numel(x)

            self.logger['loss'].append(mse.item())
            bar.set_postfix_str(f'loss={Decimal(mse.item()):.5E}.')

            self.normalize_cores_()

        return self

    def normalize_cores_(self):
        for d in range(self.order):
            norm = torch.norm(self.cores[d])
            self.cores[d].data = self.cores[d].data / norm
            self.cores[d-1].data = self.cores[d-1].data * norm

