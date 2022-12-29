# Author: Or Tal
import itertools
from typing import Union, List
import omegaconf
import torch
import torch.nn as nn


_supported_optimizers = {
    'adam': torch.optim.Adam
}


def _get_single_optimizer(cfg: omegaconf.DictConfig, models: Union[nn.Module, List[nn.Module]]):
    """
    this function initializes a single optimizer for given model / models list
    """
    assert cfg.solver.optim.optimizer.lower() in _supported_optimizers.keys(), "unsupported optimizer was given - see base_builder.py for supported dict of optimizers"
    assert isinstance(models, list) or isinstance(models, nn.Module)
    if isinstance(models, list):
        params = itertools.chain(*[m.parameters() for m in models])
    else:
        params = models.parameters()

    return _supported_optimizers[cfg.solver.optim.optimizer.lower()](params, **getattr(cfg.solver.optim, cfg.solver.optim.optimizer.lower()))


def optimizer_factory(cfg: omegaconf.DictConfig, models: Union[nn.Module, List[nn.Module], List[List[nn.Module]]]):
    if cfg.solver.optim.optimizer.lower() not in _supported_optimizers.keys():
        raise ValueError(f"unsupported optimizer was given - {cfg.solver.optim.optimizer}")
    if isinstance(models, nn.Module):
        return _get_single_optimizer(cfg, models)
    elif isinstance(models, list):
        return [_get_single_optimizer(cfg, m) for m in models]