# Author: Or Tal
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..')) 
from typing import Tuple, Union
import omegaconf
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import T_co, DistributedSampler
from .noiser import Noiser
from .simple_audio_dataset import SimpleAudioDataset
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


class NoisyDataset(Dataset):

    def __init__(self, dataset_config: omegaconf.DictConfig, partition_key_for_json_file: str, include_path=False):
        """
        :param path_to_json: dataset configuration
        :param length: max sequence length
        :param stride: stride used between two sequential audio sequences
        :param pad: pad the end of the sequence with zeros
        :param sample_rate: the signals sampling rate
        :param with_path: if True, returns path to file in addition to the signal
        """
        self.with_path = include_path
        self.audio_set = SimpleAudioDataset(dataset_config, partition_key_for_json_file, ignore_length=True, include_path=True)
        self.vad_map: dict = np.load(dataset_config.vad_file, allow_pickle=True).tolist()

    def __len__(self):
        return len(self.audio_set)

    def __getitem__(self, index) -> T_co:
        clean, clean_path = self.audio_set[index]
        #TODO: remove limitation of length?
        clean = clean[..., :min(clean.shape[-1], int(16000 * 2.5))]

        if self.with_path:
            return clean, torch.tensor(self.vad_map[clean_path.split("/")[-1]]['vad_predictions']), clean_path
        return clean, torch.tensor(self.vad_map[clean_path.split("/")[-1]]['vad_predictions'])


class NoisyLoader(DataLoader):

    def __init__(self, dataset: Dataset[T_co], dataset_config, include_clean=False, **kwargs):
        super().__init__(**kwargs, dataset=dataset, collate_fn=self.collate_func)
        self.noiser = Noiser(dataset_config)
        self.include_clean = include_clean

    def collate_func(self, batch):
        clean_sigs, vad_map = pad_sequence([b[0].flatten() for b in batch], batch_first=True).unsqueeze(1), \
                              pad_sequence([b[1] for b in batch], batch_first=True, padding_value=False)
        # clean_sigs, vad_map = pad_sequence([b[0].flatten() for b in batch], batch_first=True).unsqueeze(1), [b[1] for b in batch]

        if len(batch[0]) == 3:  # with path = loop over test set, evaluate different metrics
            noisy_sigs, clean_sigs = self.noiser(clean_sigs, is_test=True)
            return noisy_sigs, clean_sigs, vad_map, [b[2] for b in batch]
        elif self.include_clean:
            noisy_sigs, clean_sigs = self.noiser(clean_sigs, is_test=True)
            return noisy_sigs, clean_sigs, vad_map
        noisy_sigs = self.noiser(clean_sigs)
        return noisy_sigs, vad_map
