from itertools import product
from ._ssse_base_explorer import SsseBaseExplorer
# from magma.utils.cluster import get_slurm_partition


@SsseBaseExplorer
def explorer(launcher):
    # partition = get_slurm_partition()
    # launcher.slurm_(gpus=8, partition='learnlab', cpus_per_gpu=1)
    launcher.slurm_(gpus=4, partition='devlab', cpus_per_gpu=1, time=2880, comment='re_init_exps')
    # launcher.slurm_(gpus=4, partition='devlab', cpus_per_gpu=1, time=2880, comment='re_init_exps')
    launcher.bind_({
        'solver': 'solver_default',
        # 'dset.dataloader.batch_size': 128,
        'dset.dataloader.batch_size': 64,
        # 'dset.dataloader.batch_size': 96,
        'solver.optim.epochs': 400,
        'model.model_class_name': 'se_dual_ae_joint_enc',
        'model.encoder_model': 'demucs_joint_encoder',
        'solver.solver': 'se',
        'loss.loss_name': 'se_loss',
        'loss.reconstruction_factor' : 1,
        'loss.contrastive_factor' : 1,
        'loss.energy_factor' : 0.01,
        'loss.noise_regularization_factor' : 10,
        'model.include_quantizer': True,
        'model.quantizer_name': 'rvq',
        'label': 'v_0_1',
        'loss.include_contrastive': True,
        'dset.sample_from_gaussian': True
        # 'dset.sample_from_gaussian': True,
    })

    # with launcher.job_array():
    #     sub = launcher.bind({'dset.sample_from_gaussian': True, 'wandb.name': f"v0_02_joint_rvq", 'model.include_quantizer': True})
    #     # sub = launcher.bind({'loss.include_contrastive': True, 'dset.sample_from_gaussian': True, 
    #     # 'wandb.name': f"v0_01_joint_rvq", 'model.include_quantizer': True}) 
    #     # for cont, gaussian, rvq in product([True, False], [True, False], [True, False]):
    #     #     cont = True
    #     #     sub({'loss.include_contrastive': cont, 
    #     #         'dset.sample_from_gaussian': gaussian, 
    #     #         'model.include_quantizer': rvq,
    #     #         'wandb.name': f"v0_01_joint{'_rvq' if rvq else ''}_c_{int(cont)}_g_{int(gaussian)}",
    #     #     })
    #     for gaussian, rvq in product([True, False], [True, False]):
    #         sub({'dset.sample_from_gaussian': gaussian, 
    #              'model.include_quantizer': rvq,
    #              'wandb.name': f"v0_02_joint{'_rvq' if rvq else ''}_c_1_g_{int(gaussian)}",
    #         })
    #     # for cont, rvq in product([True, False], [True, False]):
    #     #     sub({'loss.include_contrastive': cont,
    #     #         'model.include_quantizer': rvq,
    #     #         'wandb.name': f"naive_loss_joint{'_rvq' if rvq else ''}_c_{int(cont)}_g_1",
    #     #     })

    with launcher.job_array():
        sub = launcher.bind({
            'wandb.name': f"v0_02_joint_rvq",
            'model.include_quantizer': True,
            'loss.include_contrastive': True,
            'loss.include_energy_loss': True
        })
        for quant, cont, energy in product([True, False], [True, False], [True, False]):
            sub({
                'wandb.name': f"v0_03_rvq_{int(quant)}_c_{int(cont)}_g_1_e_{int(energy)}",
                'model.include_quantizer': quant,
                'loss.include_energy_loss': energy, 
                'loss.include_contrastive': cont
            })