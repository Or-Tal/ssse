# Author: Or Tal
import flashy
import omegaconf
from torch.utils.data.dataloader import DataLoader
from .noisy_dataset import NoisyDataset, NoisyLoader

_supported_datasets = {
    'noisy_dataset': NoisyDataset,
}

_supported_dataloaders = {
    'noisy_loader': NoisyLoader,
}


def dataloaders_factory(cfg: omegaconf.DictConfig):
    tr_dset = dataset_factory(cfg.dset, 'tr', include_path=False)
    cv_dset = dataset_factory(cfg.dset, 'cv', include_path=False)
    tt_dset = dataset_factory(cfg.dset, 'tt', include_path=True)
    kwargs = {k: v for k, v in cfg.dset.dataloader.items()}
    kwargs["klass"] = _supported_dataloaders[cfg.dset.dataloader_class] if cfg.dset.dataloader_class in \
                      _supported_dataloaders.keys() else DataLoader
    return {
        k: flashy.distrib.loader(v, cfg.dset, include_clean=k in {'train', 'generate'}, **kwargs)
            for k, v in zip(['train', 'valid', 'evaluate', 'generate'], [tr_dset, cv_dset, tt_dset, tt_dset])
    }

def dataset_factory(dataset_cfg: omegaconf.DictConfig, partition_key_for_json_file, include_path=False):
    dataset_class = dataset_cfg.dataset_class
    if dataset_class not in _supported_datasets.keys():
        raise ValueError(f"unsupported dataset class: {dataset_class}")
    return _supported_datasets[dataset_class](dataset_cfg, partition_key_for_json_file, include_path)
