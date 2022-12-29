import logging
import os
from pathlib import Path
import sys
import typing as tp

from dora import git_save, hydra_main, XP
import flashy
import hydra
import omegaconf
from magma.utils.cluster import get_dora_dir, get_slurm_parameters
from solvers.solver_factory import solver_factory


logger = logging.getLogger(__name__)


def resolve_config_dset_paths(cfg):
    """Enable Dora to load manifest from git clone repository.
    """
    # manifest files for the different splits
    logger.info(f"cwd: {os.getcwd()}")
    for key, value in cfg.dset.items():
        if key in {'tr', 'cv', 'tt', 'gen'}:
            cfg.dset[key] = git_save.to_absolute_path(value)


def get_solver(cfg):
    # Convert batch size to batch size for each GPU
    assert cfg.dset.dataloader.batch_size % flashy.distrib.world_size() == 0
    cfg.dset.dataloader.batch_size //= flashy.distrib.world_size()
    # for split in ['train', 'valid', 'evaluate', 'generate']:
    #     if hasattr(cfg.dset, 'split') and cfg.dset.split.batch_size:
    #         assert cfg.dset[split].batch_size % flashy.distrib.world_size() == 0
    #         cfg.dataset[split].batch_size //= flashy.distrib.world_size()
    # resolve_config_dset_paths(cfg)
    solver = solver_factory(cfg)
    return solver


def get_solver_from_xp(xp: XP, override_cfg: tp.Optional[dict] = None,
                       restore: bool = True):
    """Given a XP, return the Solver object.
    Args:
        xp (XP): Dora experiment for which to retrieve the solver.
        override_cfg (dict or None): if not None, should be a dict used to
            override some values in the config of `xp`. This will not impact
            the XP signature or folder. The format is different
            than the one used in Dora grids, nested keys should actually be nested dicts,
            not flattened, e.g. `{'optim': {'batch_size': 32}}`.
        restore (bool): if `True` (the default), restore state from the last checkpoint.
    """
    logger.info(f"Loading solver from XP {xp.sig}. "
                f"Overrides used: {xp.argv}")
    cfg = xp.cfg
    if override_cfg is not None:
        cfg = omegaconf.OmegaConf.merge(cfg, omegaconf.DictConfig(override_cfg))
    try:
        with xp.enter():
            solver = get_solver(cfg)
            if restore:
                solver.restore()
        return solver
    finally:
        hydra.core.global_hydra.GlobalHydra.instance().clear()


def get_solver_from_sig(sig: str, override_cfg: tp.Optional[dict] = None,
                        restore: bool = True):
    """Return Solver object from Dora signature, i.e. to play with it from a notebook.
    See `get_solver_from_xp` for more information.
    """
    xp = main.get_xp_from_sig(sig)
    return get_solver_from_xp(xp, override_cfg, restore)


def init_seed(cfg):
    import numpy as np
    import torch
    import random
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    # torch also initialize cuda seed if available
    torch.manual_seed(cfg.seed)


@hydra_main(config_path='../config', config_name='main_config', version_base='1.1')
def main(cfg):
    init_seed(cfg)
    # Setup logging both to XP specific folder, and to stderr.
    log_name = 'generate.log.{rank}' if cfg.generate_only else 'solver.log.{rank}'
    flashy.setup_logging(level=str(cfg.logging.level).upper(), log_name=log_name)
    # Initialize distributed training, no need to specify anything when using Dora.
    flashy.distrib.init()
    solver = get_solver(cfg)
    if cfg.show:
        solver.show()
        return
    elif cfg.generate_only:
        solver.run_generate()
        return
    else:
        return solver.run()


if '_DORA_TEST_PATH' in os.environ:
    main.dora.dir = Path(os.environ['_DORA_TEST_PATH'])
    main.dora.shared = None
else:
    # TODO: try to avoid hardcoding here...
    main.dora.dir = get_dora_dir('')
    main._base_cfg.slurm = get_slurm_parameters(main._base_cfg.slurm)

if main.dora.shared is not None and not os.access(main.dora.shared, os.R_OK):
    print("No read permission on dora.shared folder, ignoring it.", file=sys.stderr)
    main.dora.shared = None


if __name__ == '__main__':
    print("")
    main()