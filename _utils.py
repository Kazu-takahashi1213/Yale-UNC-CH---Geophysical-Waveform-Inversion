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
