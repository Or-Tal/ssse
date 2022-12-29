from itertools import product
from .ssse_base_explorer import SsseBaseExplorer
# from magma.utils.cluster import get_slurm_partition


@SsseBaseExplorer
def explorer(launcher):
    # partition = get_slurm_partition()
    launcher.slurm_(gpus=8, partition='devfair')
    launcher.bind_({
        'solver': 'solver_default',
        'solver.optim.epochs': 500,
    })
    with launcher.job_array():
        sub = launcher.bind({'loss.include_regularization': True, 'loss.include_contrastive': True}) 
        for reg, cont in product([True, False], [True, False]):
            sub({'loss.include_regularization': reg, 'loss.include_contrastive': cont})
