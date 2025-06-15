
import numpy as np
import torch
import random

def randaugment_pair(x, y, ops=2, magnitude=9):
    for _ in range(ops):
        op = random.choice(['flip','shift','noise'])
        if op == 'flip':
            x = x[..., ::-1].copy()
            y = y[..., ::-1].copy()
        elif op == 'shift':
            shift = int(np.random.uniform(-magnitude, magnitude))
            x = np.roll(x, shift, axis=-1)
            y = np.roll(y, shift, axis=-1)
        elif op == 'noise':
            x = x + np.random.normal(0, magnitude/30.0, size=x.shape)
    return x, y

def mixup(x, y, alpha):
    if alpha <= 0:
        return x, y
    lam = np.random.beta(alpha, alpha)
    perm = torch.randperm(x.size(0), device=x.device)
    x2, y2 = x[perm], y[perm]
    x = lam * x + (1 - lam) * x2
    y = lam * y + (1 - lam) * y2
    return x, y

def cutmix(x, y, alpha):
    if alpha <= 0:
        return x, y
    lam = np.random.beta(alpha, alpha)
    perm = torch.randperm(x.size(0), device=x.device)
    x2, y2 = x[perm], y[perm]
    B, C, H, W = x.shape
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    w = int(W * np.sqrt(1 - lam))
    h = int(H * np.sqrt(1 - lam))
    x1 = np.clip(cx - w//2, 0, W)
    x2r = np.clip(cx + w//2, 0, W)
    y1 = np.clip(cy - h//2, 0, H)
    y2r = np.clip(cy + h//2, 0, H)
    x[:, :, y1:y2r, x1:x2r] = x2[:, :, y1:y2r, x1:x2r]
    y[:, :, y1:y2r, x1:x2r] = y2[:, :, y1:y2r, x1:x2r]
    return x, y
