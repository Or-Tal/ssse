# Author: Or Tal

import omegaconf
import torch.nn as nn

from ssse.losses.naive_se_loss import NaiveSELoss
from .se_loss import SELoss, SupSELoss

_supported_losses = {
    'se_loss': SELoss,
    'sup_se_loss': SupSELoss,
    'noive_se_loss': NaiveSELoss,
}


def loss_factory(cfg: omegaconf.DictConfig) -> nn.Module:
    if cfg.loss.loss_name.lower() in _supported_losses.keys():
        return _supported_losses[cfg.loss.loss_name.lower()](cfg.loss, device=cfg.device)
    else:
        raise ValueError(f"invalid model class: {cfg.model.model_class_name}")