from itertools import product
from ._ssse_base_explorer import SsseBaseExplorer
# from magma.utils.cluster import get_slurm_partition


@SsseBaseExplorer
def explorer(launcher):
    # partition = get_slurm_partition()
    launcher.slurm_(gpus=8, partition='devlab')
    launcher.bind_({
        'solver': 'solver_default',
        'solver.optim.epochs': 400,
    })

    with launcher.job_array():
        sub = launcher.bind({'loss.include_contrastive': True, 'dset.sample_from_gaussian': False}) 
        for cont, gaussian in product([True, False], [True, False]):
            sub({'loss.include_contrastive': cont, 'dset.sample_from_gaussian': gaussian})
