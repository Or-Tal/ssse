# Author: Or Tal
import json
import numpy as np
import torch
import torch.nn as nn
import torchaudio
from dataclasses import dataclass


@dataclass
class NoiseLen:
    path: str
    length: int


class Noiser(nn.Module):

    def __init__(self, dataset_config):
        super().__init__()
        self.cfg = dataset_config
        with open(dataset_config.noises, "r") as f:
            self.noises = {k: [NoiseLen(*t) for t in v] for k, v in json.load(f).items()}

        self.noises_for_test = dataset_config.noises_for_test
        self.noises_for_train = [k for k in self.noises.keys() if k not in self.noises_for_test]
        self.snr_range = torch.arange(*dataset_config.snr_range)

        self.sample_from_gaussian = hasattr(dataset_config, "sample_from_gaussian") and \
                                    dataset_config.sample_from_gaussian

    def pow_sq(self, x, eps=1e-5):
        return torch.sum(x ** 2, dim=-1) + eps

    def snr(self, signal, noise, eps=1e-5):
        snr_value = (self.pow_sq(signal, eps)) / (self.pow_sq(noise, eps))
        return 10 * torch.log10(snr_value)

    @staticmethod
    def get_audio(path_to_audio, audio_length, frame_length_to_read):
        frame_offset = np.random.randint(0, audio_length - frame_length_to_read)
        out, _ = torchaudio.load(str(path_to_audio), frame_offset=frame_offset, num_frames=frame_length_to_read)
        return out

    def sample_noise(self, is_test, frame_length_to_read):
        if self.sample_from_gaussian:
            return torch.normal(0, 1, size=(frame_length_to_read,))
        noise_type = np.random.choice(self.noises_for_test if is_test else self.noises_for_train)
        t = np.random.choice(self.noises[noise_type])
        return self.get_audio(t.path, t.length, frame_length_to_read)

    def sample_batch_of_noises(self, input_shape, is_test):
        noises = [self.sample_noise(is_test, input_shape[-1]) for _ in range(input_shape[0])]
        return torch.stack(noises, dim=0)

    def forward(self, clean_audio, is_test=False):

        # sample noise uniformly from noise sources
        noises = self.sample_batch_of_noises(clean_audio.shape, is_test)

        # add channel dim if such exists
        if len(clean_audio.shape) == 3 and len(noises.shape) == 2:
            noises = noises.unsqueeze(1)

        # randomly sample snrs and scale noise to modify the snr;
        # SNR = 10 * log_10(P_signal / P_noise)
        snrs = self.snr(clean_audio, noises).flatten()
        randomly_sampled_snrs = torch.tensor(np.random.choice(self.snr_range, clean_audio.shape[0]))
        factors = ((10 ** (snrs / 10)) * 1 / (10 ** (randomly_sampled_snrs / 10))) ** 0.5
        noises = noises.permute(2, 1, 0)
        noises = noises * factors.expand_as(noises)
        noises = noises.permute(2, 1, 0)
        if is_test:
            return clean_audio + noises, clean_audio
        return clean_audio + noises