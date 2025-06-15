
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
    teacher_model='vit_base_patch16_224',
    temperature=4.0,
    alpha=0.5,
)
cfg.phys_weight = 0.1
cfg.multi_scales = [1, 2, 4]
