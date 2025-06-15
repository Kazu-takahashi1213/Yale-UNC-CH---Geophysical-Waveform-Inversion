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
