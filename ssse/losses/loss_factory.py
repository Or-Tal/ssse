# Author: Or Tal

import omegaconf
import torch.nn as nn

from .se_loss import SELoss

_supported_losses = {
    'se_loss': SELoss,
}


def loss_factory(cfg: omegaconf.DictConfig) -> nn.Module:
    if cfg.loss.loss_name.lower() in _supported_losses.keys():
        return _supported_losses[cfg.loss.loss_name.lower()](cfg.loss, device=cfg.device)
    else:
        raise ValueError(f"invalid model class: {cfg.model.model_class_name}")