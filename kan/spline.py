import torch
import torch.nn as nn
import numpy as np

def B_batch(x, grid, k=0, extend=True, device='cpu'):
    def extend_grid(grid, k_extend=0):
        h = (grid[:, [-1]] - grid[:, [0]]) / (grid.shape[1] - 1)
        for i in range(k_extend):
            grid = torch.cat([grid[:, [0]] - h, grid], dim=1)
            grid = torch.cat([grid, grid[:, [-1]] + h], dim=1)
        grid = grid.to(device)
        return grid

    if extend:
        grid = extend_grid(grid, k_extend=k)

    grid = grid.unsqueeze(dim=2).to(device)
    x = x.unsqueeze(dim=1).to(device)

    if k == 0:
        value = (x >= grid[:, :-1]) * (x < grid[:, 1:])
    else:
        B_km1 = B_batch(x[:, 0], grid=grid[:, :, 0], k=k - 1, extend=False, device=device)
        value = (x - grid[:, :-(k + 1)]) / (grid[:, k:-1] - grid[:, :-(k + 1)]) * B_km1[:, :-1] + (
                    grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:(-k)]) * B_km1[:, 1:]
    return value

def coef2curve(x_eval, grid, coef, k, device="cpu"):
    if coef.dtype != x_eval.dtype:
        coef = coef.to(x_eval.dtype)
    B = B_batch(x_eval, grid, k, device=device)
    
    if B.shape[1] != coef.shape[1]:
        raise ValueError(f"Number of coefficients ({coef.shape[1]}) does not match the number of B-spline basis functions ({B.shape[1]}).")
    
    y_eval = torch.einsum('ij,ijk->ik', coef, B)
    
    return y_eval

def curve2coef(x_eval, y_eval, grid, k, device="cpu"):
    mat = B_batch(x_eval, grid, k, device=device).permute(0, 2, 1)
    coef = torch.linalg.lstsq(mat.to(device), y_eval.unsqueeze(dim=2).to(device),
                              driver='gelsy' if device == 'cpu' else 'gels').solution[:, :, 0]
    return coef.to(device)
