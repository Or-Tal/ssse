from itertools import product
from ._ssse_base_explorer import SsseBaseExplorer
# from magma.utils.cluster import get_slurm_partition


@SsseBaseExplorer
def explorer(launcher):
    # partition = get_slurm_partition()
    # launcher.slurm_(gpus=8, partition='learnlab', cpus_per_gpu=1)
    launcher.slurm_(gpus=4, partition='devlab', cpus_per_gpu=1, time=2880, comment='exp_joint_v0')
    launcher.bind_({
        'solver': 'solver_default',
        'dset.dataloader.batch_size': 96,
        'solver.optim.epochs': 400,
        'model.model_class_name': 'se_dual_ae_joint_enc',
        'solver.solver': 'se',
        'loss.loss_name': 'se_loss',
        'loss.reconstruction_factor' : 1,
        'loss.contrastive_factor' : 10,
        'loss.noise_regularization_factor' : 1,
    })

    with launcher.job_array():
        sub = launcher.bind({'loss.include_contrastive': True, 'dset.sample_from_gaussian': True, 'wandb.name': f"joint_"}) 
        for cont, gaussian in product([True, False], [True, False]):
            sub({'loss.include_contrastive': cont, 
                'dset.sample_from_gaussian': gaussian, 
                'wandb.name': f"joint_c{int(cont)}_g{int(gaussian)}",
            })