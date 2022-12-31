# Author: Or Tal
import omegaconf
from .sup_se_solver import SupSESolver
from .se_solver import SESolver

_supported_solvers = {
    "se": SESolver,
    "sup_se": SupSESolver,
}

def solver_factory(cfg: omegaconf.DictConfig):
    assert cfg.solver.solver.lower() in _supported_solvers.keys(), f"unsupported solver was given - {cfg.solver.solver.lower()}"
    return _supported_solvers[cfg.solver.solver.lower()](cfg)