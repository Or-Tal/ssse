from itertools import product
from ._ssse_base_explorer import SsseBaseExplorer
# from magma.utils.cluster import get_slurm_partition


@SsseBaseExplorer
def explorer(launcher):
    # partition = get_slurm_partition()
    # launcher.slurm_(gpus=8, partition='learnlab', cpus_per_gpu=1)
    launcher.slurm_(gpus=4, partition='learnlab', cpus_per_gpu=1, time=2880, comment='')
    launcher.bind_({
        'solver': 'solver_default',
        'dset.dataloader.batch_size': 96,
        'solver.optim.epochs': 400,
        'loss.include_contrastive': True,
        'dset.sample_from_gaussian': True,
        'solver.solver': 'se',
        'loss.loss_name': 'se_loss',
        'loss.reconstruction_factor': 1,
    })

    with launcher.job_array():
        sub = launcher.bind({
            'loss.contrastive_factor': 1,
            'loss.noise_regularization_factor': 1,
            'wandb.name': f"hparam_search"}) 
        for c_factor, reg_factor, lr in product([0.05, 0.1, 0.3, 1, 2, 5, 10], [0.05, 0.1, 0.3, 1, 2, 5, 10], [1e-3, 7e-4, 3e-4, 1e-4, 7e-5, 3e-5, 1e-5]):
            sub({'loss.contrastive_factor': c_factor, 
                'loss.noise_regularization_factor': reg_factor, 
                'wandb.name': f"hparam_c_{str(c_factor).replace('.', '_')}_reg_{str(reg_factor).replace('.', '_')}_lr_{str(lr).replace('.', '_')}",
                })