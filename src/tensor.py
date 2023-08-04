import torch
from typing import List, Optional


def full_tr(cores, weight=None):
    dim = len(cores)
    if weight is not None:
        x = torch.einsum('imn, n-> imn', cores[0], weight[0])
    else:
        x = cores[0]
    for i in range(1, dim):
        if weight is not None:
            core = torch.einsum('imn, n-> imn', cores[i], weight[i])
        else:
            core = cores[i]
        x = torch.einsum('...mn, jnk-> ...jmk', x, core)

    x = torch.einsum('...mm-> ...', x)
    return x


def full_tr_idx(cores, idx: torch.Tensor, weight: Optional[List[torch.Tensor]]=None):
    num, dim = idx.shape
    if weight is not None:
        tr_val = torch.einsum('imn, n-> imn', cores[0][idx[:, 0], :, :], weight[0])
    else:
        tr_val = cores[0][idx[:, 0], :, :]
    for d in range(1, dim):
        if weight is not None:
            core = torch.einsum('imn, n-> imn', cores[d][idx[:, d], :, :], weight[d])
        else:
            core = cores[d][idx[:, d], :, :]
        tr_val = torch.einsum('imn, ink-> imk', tr_val, core)
    tr_val = torch.einsum('ikk-> i', tr_val)

    return tr_val


def get_sub_chain(cores, weight=None):
    dim = len(cores)

    right_chain = get_right_chain(cores, weight)
    left_chain = get_left_chain(cores, weight)

    out = []
    for d in range(dim):
        if d == 0:
            foo = left_chain[0]
        elif d == dim - 1:
            foo = right_chain[-1]
        else:
            foo = contract_left_right_chain(left_chain[d], right_chain[d-1])
        out.append(foo)

    return out


def contract_left_right_chain(left_chain, right_chain):
    dim = (left_chain.ndim - 2) + (right_chain.ndim - 2) + 1
    d = right_chain.ndim - 2
    out = torch.tensordot(left_chain, right_chain, dims=([-1], [-2]))
    out = torch.moveaxis(out, dim - d - 1, -2)

    return out


def get_right_chain(cores, weight=None):
    dim = len(cores)
    # right sweep
    if weight is not None:
        core = torch.einsum('imn, n-> imn', cores[0], weight[0])
    else:
        core = cores[0]
    right_chain = [core]  # Q_1, Q_{1, 2}, ..., Q_{1, ..., D-1}
    for i in range(1, dim-1):
        if weight is not None:
            core = torch.einsum('imn, n-> imn', cores[i], weight[i])
        else:
            core = cores[i]
        foo = torch.einsum('...mn, bnk-> ...bmk', right_chain[-1], core)
        right_chain.append(foo)
        
    return right_chain
        

def get_left_chain(cores, weight=None):
    dim = len(cores)
    # left sweep
    if weight is not None:
        core = torch.einsum('imn, n-> imn', cores[-1], weight[-1])
    else:
        core = cores[-1]
    left_chain = [core]  # Q_{2, ..., D}, Q_{3, ..., D}, ..., Q_D
    for i in range(dim-2, 0, -1):
        if weight is not None:
            core = torch.einsum('imn, n-> imn', cores[i], weight[i])
        else:
            core = cores[i]
        foo = torch.einsum('amn, ...nk-> a...mk', core, left_chain[0])
        left_chain.insert(0, foo)

    return left_chain


def get_left_chain_idx(idx, cores, weight=None):
    dim = len(cores)
    # left sweep
    idx_d = idx[:, -1]
    if weight is not None:
        core = torch.einsum('imn, n-> imn', cores[-1][idx_d, :, :], weight[-1])
    else:
        core = cores[-1][idx_d, :, :]
    left_chain = [core]  # Q_{2, ..., D}, Q_{3, ..., D}, ..., Q_D
    for i in range(dim-2, 0, -1):
        idx_d = idx[:, i]
        if weight is not None:
            core = torch.einsum('imn, n-> imn', cores[i][idx_d, :, :], weight[i])
        else:
            core = cores[i][idx_d, :, :]
        foo = torch.einsum('imn, ink-> imk', core, left_chain[0])
        left_chain.insert(0, foo)

    return left_chain


if __name__ == '__main__':
    cores = []
    sz = [3, 4, 5, 6]
    bond_dim = [7, 8, 9, 10]
    for i in range(len(sz)):
        cores.append(torch.randn(sz[i], bond_dim[i-1], bond_dim[i]))

    x = full_tr(cores)
    x_ = torch.einsum('amn, bni, cij, djm-> abcd', *cores)
    print(torch.norm(x - x_) / torch.norm(x))

    sub_chain = get_sub_chain(cores)
    for q in sub_chain:
        print(q.shape)
