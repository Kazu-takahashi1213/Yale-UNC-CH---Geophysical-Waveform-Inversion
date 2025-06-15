"""Export training helpers for Kaggle.

Executing this script writes auxiliary modules (_cfg.py, _dataset.py, etc.)
so that the training code can run from a single file upload.
"""
import os
import textwrap

MODULES = {
    '_cfg.py': '''
from types import SimpleNamespace
import torch

cfg= SimpleNamespace()
cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg.local_rank = 0
cfg.seed = 123
cfg.subsample = None

cfg.backbone1 = "caformer_b36.sail_in22k_ft_in1k"
cfg.backbone = cfg.backbone1  # alias used by scripts
cfg.batch_size = 8
cfg.batch_size_val = 16
cfg.epochs = 10

cfg.backbone2 = "convnext_small.fb_in22k_ft_in1k"
cfg.ema = True
cfg.ema_decay = 0.99

cfg.early_stopping = {"patience": 3, "streak": 0}
cfg.logging_steps = 100

cfg.aug = SimpleNamespace(
    randaugment_ops=2,
    randaugment_magnitude=9,
    mixup_alpha=0.8,
    cutmix_alpha=1.0,
    cutmix_prob=1.0,
    switch_prob=0.5,
    label_smoothing=0.1,
)
cfg.distill = SimpleNamespace(
    teacher_models=[
        'vit_large_patch16_224',
        'vit_base_patch16_224',
    ],
    temperature=4.0,
    alpha=0.5,
)
cfg.distill.teacher_model = cfg.distill.teacher_models[0]
cfg.postprocess = SimpleNamespace(
    diffusion_steps=3,
    diffusion_sigma=0.1,
)
cfg.phys_weight = 0.1
cfg.multi_scales = [1, 2, 4]
''',
    '_dataset.py': '''
import os
import glob

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from _aug import randaugment_pair

class CustomDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        cfg,
        mode = "train",
    ):
        self.cfg = cfg
        self.mode = mode

        self.data, self.labels, self.records = self.load_metadata()

    def load_metadata(self, ):

        # Select rows
        df= pd.read_csv("/kaggle/input/openfwi-preprocessed-72x72/folds.csv")
        if self.cfg.subsample is not None:
            df= df.groupby(["dataset", "fold"]).head(self.cfg.subsample)

        if self.mode == "train":
            df= df[df["fold"] != 0]
        else:
            df= df[df["fold"] == 0]


        data = []
        labels = []
        records = []
        mmap_mode = "r"

        for i,row in tqdm(df.iterrows(), total=len(df)):
            d_path = "/kaggle/input/openfwi-preprocessed-72x72/" + row['dataset'] + "/" + row['data']
            l_path = "/kaggle/input/openfwi-preprocessed-72x72/" + row['dataset'] + "/" + row['label']
            records.append(row.to_dict())
            data.append(np.load(d_path, mmap_mode=mmap_mode))
            labels.append(np.load(l_path, mmap_mode=mmap_mode))

        return data, labels, records

    def __len__(self):
        return len(self.records)*500

    def __getitem__(self, index):
        file_idx = index // 500
        data_idx = index % 500
        x = self.data[file_idx][data_idx]
        y = self.labels[file_idx][data_idx]
        if self.mode == "train":
            x, y = randaugment_pair(x, y, self.cfg.aug.randaugment_ops, self.cfg.aug.randaugment_magnitude)
        x = torch.tensor(x).float()
        y = torch.tensor(y).float()
        return x, y
''',
    '_aug.py': '''
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
''',
    '_model.py': '''
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import timm

class ModelEMA(nn.Module):
    """Exponential Moving Average model wrapper."""
    def __init__(self, model, decay=0.99, device=None):
        super().__init__()
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device
        if self.device is not None:
            self.module.to(device=self.device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)

class Net(nn.Module):
    """Simple segmentation network with a timm backbone."""
    def __init__(self, backbone: str):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0, in_chans=5, features_only=True)
        ch = self.backbone.feature_info.channels()[-1]
        self.head = nn.Sequential(
            nn.Conv2d(ch, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(128, 1, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)[-1]
        out = self.head(feat)
        out = F.interpolate(out, size=(70, 70), mode='bilinear', align_corners=False)
        return out
''',
    '_utils.py': '''
import datetime
import torch

def format_time(elapsed: float) -> str:
    """Format seconds to hh:mm:ss."""
    elapsed_rounded = int(round(elapsed))
    return str(datetime.timedelta(seconds=elapsed_rounded))

def diffusion_smoothing(x, steps=3, sigma=0.1):
    """Simple diffusion-like smoothing used as a post process."""
    for _ in range(steps):
        noise = torch.randn_like(x) * sigma
        x = x + noise
        x = torch.nn.functional.avg_pool2d(x, 3, stride=1, padding=1)
    return x
''',
    '_train.py': '''
import os
import time
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler

import torch.distributed as dist
from torch.utils.data import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

from _cfg import cfg
from _dataset import CustomDataset
from _model import ModelEMA, Net
from _utils import format_time, diffusion_smoothing
from _aug import mixup, cutmix


import torch.nn.functional as F
import timm

def gradient_loss(pred, target):
    dx_pred = pred[..., 1:, :] - pred[..., :-1, :]
    dy_pred = pred[..., :, 1:] - pred[..., :, :-1]
    dx_tgt = target[..., 1:, :] - target[..., :-1, :]
    dy_tgt = target[..., :, 1:] - target[..., :, :-1]
    return (dx_pred - dx_tgt).abs().mean() + (dy_pred - dy_tgt).abs().mean()

def multi_scale_loss(pred, target, scales):
    loss = F.l1_loss(pred, target)
    for s in scales[1:]:
        p = F.avg_pool2d(pred, kernel_size=s)
        t = F.avg_pool2d(target, kernel_size=s)
        loss = loss + F.l1_loss(p, t)
    return loss / len(scales)

def set_seed(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def setup(rank, world_size):
    torch.cuda.set_device(rank)
    if world_size > 1:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    return

def cleanup():
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    return

def main(cfg):

    # ========== Datasets / Dataloaders ==========
    if cfg.local_rank == 0:
        print("="*25)
        print("Loading data..")
    train_ds = CustomDataset(cfg=cfg, mode="train")
    if cfg.world_size > 1:
        sampler = DistributedSampler(train_ds, num_replicas=cfg.world_size, rank=cfg.local_rank)
        shuffle = False
    else:
        sampler = None
        shuffle = True
    train_dl = torch.utils.data.DataLoader(
        train_ds,
        sampler=sampler,
        shuffle=shuffle,
        batch_size=cfg.batch_size,
        num_workers=4,
    )

    valid_ds = CustomDataset(cfg=cfg, mode="valid")
    if cfg.world_size > 1:
        sampler = DistributedSampler(valid_ds, num_replicas=cfg.world_size, rank=cfg.local_rank)
        shuffle = False
    else:
        sampler = None
        shuffle = False
    valid_dl = torch.utils.data.DataLoader(
        valid_ds,
        sampler=sampler,
        shuffle=shuffle,
        batch_size=cfg.batch_size_val,
        num_workers=4,
    )

    # ========== Model / Optim ==========
    model = Net(backbone=cfg.backbone)
    model = model.to(cfg.local_rank)

    teachers = []
    for tb in cfg.distill.teacher_models:
        t_net = Net(backbone=tb)
        t_net = t_net.to(cfg.local_rank)
        t_net.eval()
        teachers.append(t_net)
    if cfg.ema:
        if cfg.local_rank == 0:
            print("Initializing EMA model..")
        ema_model = ModelEMA(
            model,
            decay=cfg.ema_decay,
            device=cfg.local_rank,
        )
    else:
        ema_model = None
    if cfg.world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[cfg.local_rank],
        )

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = GradScaler()


    # ========== Training ==========
    if cfg.local_rank == 0:
        print("="*25)
        print("Give me warp {}, Mr. Sulu.".format(cfg.world_size))
        print("="*25)

    best_loss= 1_000_000
    val_loss= 1_000_000

    for epoch in range(0, cfg.epochs+1):
        if epoch != 0:
            tstart = time.time()
            if hasattr(train_dl.sampler, "set_epoch"):
                train_dl.sampler.set_epoch(epoch)

            # Train loop
            model.train()
            total_loss = []
            for i, (x, y) in enumerate(train_dl):
                x = x.to(cfg.local_rank)
                y = y.to(cfg.local_rank)
                if np.random.rand() < cfg.aug.cutmix_prob:
                    x, y = cutmix(x, y, cfg.aug.cutmix_alpha)
                else:
                    x, y = mixup(x, y, cfg.aug.mixup_alpha)

                with autocast(cfg.device.type):
                    with torch.no_grad():
                        t_outs = [t(x) for t in teachers]
                        t_logits = torch.stack(t_outs).mean(dim=0)
                    logits = model(x)
                    logits = diffusion_smoothing(
                        logits,
                        cfg.postprocess.diffusion_steps,
                        cfg.postprocess.diffusion_sigma,
                    )
                    mae = multi_scale_loss(logits, y, cfg.multi_scales)
                    distill_loss = F.mse_loss(logits, t_logits)
                    phys = gradient_loss(logits, y) * cfg.phys_weight
                    loss = ((1 - cfg.distill.alpha) * mae + cfg.distill.alpha * distill_loss / (cfg.distill.temperature ** 2) + phys)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                total_loss.append(loss.item())

                if ema_model is not None:
                    ema_model.update(model)

                if cfg.local_rank == 0 and (len(total_loss) >= cfg.logging_steps or i == 0):
                    train_loss = np.mean(total_loss)
                    total_loss = []
                    print("Epoch {}:     Train MAE: {:.2f}     Val MAE: {:.2f}     Time: {}     Step: {}/{}".format(
                        epoch,
                        train_loss,
                        val_loss,
                        format_time(time.time() - tstart),
                        i+1,
                        len(train_dl)+1,
                    ))

        # ========== Valid ==========
        model.eval()
        val_logits = []
        val_targets = []
        with torch.no_grad():
            for x, y in tqdm(valid_dl, disable=cfg.local_rank != 0):
                x = x.to(cfg.local_rank)
                y = y.to(cfg.local_rank)

                with autocast(cfg.device.type):
                    if ema_model is not None:
                        out = ema_model.module(x)
                    else:
                        out = model(x)
                    out = diffusion_smoothing(
                        out,
                        cfg.postprocess.diffusion_steps,
                        cfg.postprocess.diffusion_sigma,
                    )

                val_logits.append(out.cpu())
                val_targets.append(y.cpu())

            val_logits= torch.cat(val_logits, dim=0)
            val_targets= torch.cat(val_targets, dim=0)

            loss = multi_scale_loss(val_logits, val_targets, cfg.multi_scales).item()

        # Gather loss
        v = torch.tensor([loss], device=cfg.local_rank)
        if cfg.world_size > 1:
            torch.distributed.all_reduce(v, op=dist.ReduceOp.SUM)
            val_loss = (v[0] / cfg.world_size).item()
        else:
            val_loss = v.item()

        # ========== Weights / Early stopping ==========
        stop_train = torch.tensor([0], device=cfg.local_rank)
        if cfg.local_rank == 0:
            es= cfg.early_stopping
            if val_loss < best_loss:
                print("New best: {:.2f} -> {:.2f}".format(best_loss, val_loss))
                print("Saved weights..")
                best_loss = val_loss
                if ema_model is not None:
                    torch.save(ema_model.module.state_dict(), f'best_model_{cfg.seed}.pt')
                else:
                    torch.save(model.state_dict(), f'best_model_{cfg.seed}.pt')

                es["streak"] = 0
            else:
                es= cfg.early_stopping
                es["streak"] += 1
                if es["streak"] > es["patience"]:
                    print("Ending training (early_stopping).")
                    stop_train = torch.tensor([1], device=cfg.local_rank)

        # Exits training on all ranks
        if cfg.world_size > 1:
            dist.broadcast(stop_train, src=0)
        if stop_train.item() == 1:
            return

    return


if __name__ == "__main__":

    # GPU Specs
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    _, total = torch.cuda.mem_get_info(device=rank)

    # Init
    setup(rank, world_size)
    time.sleep(rank)
    print(f"Rank: {rank}, World size: {world_size}, GPU memory: {total / 1024**3:.2f}GB", flush=True)
    time.sleep(world_size - rank)

    # Seed
    set_seed(cfg.seed+rank)

    # Run
    cfg.local_rank= rank
    cfg.world_size= world_size
    main(cfg)
    cleanup()
'''
}

def export(path="."):
    """Write helper modules to *path*."""
    os.makedirs(path, exist_ok=True)
    for fname, text in MODULES.items():
        with open(os.path.join(path, fname), "w") as f:
            f.write(textwrap.dedent(text))
    print("Exported modules to", os.path.abspath(path))

if __name__ == "__main__":
    export()
