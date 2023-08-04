# Scalable Bayesian Tensor Ring Factorization for Multiway Data Analysis

## Description

This is the official implementation for the following paper presented in ICONIP 2023.
```
@inproceedings{tao2023scalable_btr,
  title={Scalable Bayesian Tensor Ring Factorization for Multiway Data Analysis},
  author={Tao, Zerui and Tanaka, Toshihisa and Zhao, Qibin},
  booktitle={International Conference on Neural Information Processing},
  year={2023},
  publisher={Springer International Publishing}
}
```

The code is mainly based on `Python 3.9` and `PyTorch 1.12.1`.

Run `pip install -r requirements.txt` to install all required libraries.

## Examples

1. To reproduce the experiment on rank estimation,
run `./simulation_rank.py` file.

2. Example of running continuous data completion is in `./ushcn.py`.

3. Example of running binary data completion is in `./enron.py`.
