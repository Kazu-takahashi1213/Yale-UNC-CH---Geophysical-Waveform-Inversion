# Geophysical Waveform Inversion


This repository contains helper modules for Kaggle competitions on full waveform inversion. The training loop can run either in distributed mode or on a single GPU (e.g. Kaggle). It includes several data augmentation strategies and a knowledge distillation setup.

## Running on Kaggle

1. Upload either the individual helper files or just `caformer.py` to your Kaggle Notebook or dataset.
   If you only upload `caformer.py`, run `python caformer.py` once to export
   the helper modules.

2. In a Kaggle Notebook cell run:
   ```python
   !python _train.py
   ```
   The script will automatically fall back to single GPU training when `RANK` and `WORLD_SIZE` are not set.
3. The best model weights will be saved as `best_model_<SEED>.pt` in the working directory.

Edit `_cfg.py` to adjust hyperparameters such as batch size or number of epochs.


### Features

- **RandAugment, MixUp, and CutMix** data augmentation in the training pipeline.
- **Knowledge distillation** with a configurable teacher model.
- **EMA weights** and gradient-based physical loss for smoother convergence.
- **Ensemble distillation** from multiple teachers for improved generalization.
- **Diffusion-based post-processing** to smooth predictions before evaluation.

