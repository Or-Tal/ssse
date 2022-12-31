from itertools import product
from ._ssse_base_explorer import SsseBaseExplorer
# from magma.utils.cluster import get_slurm_partition


@SsseBaseExplorer
def explorer(launcher):
    # partition = get_slurm_partition()
    # launcher.slurm_({'gpus': 8, 'partition': 'devlab', 'cpus-per-gpu': 1, 'mem-per-gpu': '1g'})
    launcher.slurm_(gpus=8, partition='devlab', cpus_per_gpu=1)
    launcher.bind_({
        'solver': 'solver_default',
        'solver.optim.epochs': 400,
    })

    with launcher.job_array():
        sub = launcher.bind({'loss.include_contrastive': True, 'dset.sample_from_gaussian': False, 'wandb.name': f"base_exp", 
                             'solver.solver': 'se', 'loss.loss_name': 'se_loss',}) 
        for pref in ["", "sup_"]:
            for cont, gaussian in product([True, False], [True, False]):
                sub({'loss.include_contrastive': cont, 
                     'dset.sample_from_gaussian': gaussian, 
                     'wandb.name': f"base_{pref}c{int(cont)}_g{int(gaussian)}",
                     'solver.solver': f'{pref}se',
                     'loss.loss_name': f'{pref}se_loss',
                     })