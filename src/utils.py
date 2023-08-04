import torch


def perm_dim(dim, d):
    perm = list(range(d + 1, dim)) + list(range(d + 1))

    return perm


def gen_mask(x, msr):
    mask = torch.zeros_like(x)
    obs_num = round(torch.numel(x) * (1. - msr))
    obs_idx = torch.randperm(torch.numel(x))[:obs_num]
    mask = mask.reshape(-1)
    mask[obs_idx] = 1.0
    mask = mask.reshape(x.shape)
    return mask


def rse_metric(x, x_hat):
    rse = torch.norm(x - x_hat, p=2) / torch.norm(x)
    return rse


def psnr_metric(x, x_hat):
    mse = torch.norm(x - x_hat, p=2) ** 2 / torch.numel(x)
    psnr = 10 * torch.log10(torch.max(x) ** 2 / mse)
    return psnr


def get_batch_data(x, idx, batch_size):
    shuffle = torch.randperm(idx.shape[0])
    x = x[shuffle]
    idx = idx[shuffle]
    x_batch = torch.split(x, split_size_or_sections=batch_size)
    idx_batch = torch.split(idx, split_size_or_sections=batch_size)

    return x_batch, idx_batch
