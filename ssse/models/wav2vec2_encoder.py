import torch.nn as nn
import torch
from fairseq.models.wav2vec.wav2vec2 import Wav2Vec2Config, Wav2Vec2Model

FT_KEY = 'x'
# FT_KEY = 'layer_results'


class Wav2vec2Encoder(nn.Module):

    def __init__(self, encoder_configuration):
        super().__init__()

        self.encoder = Wav2Vec2Model(cfg=Wav2Vec2Config(**encoder_configuration))

    def forward(self, x):
        return self.encoder(x.squeeze(1), padding_mask=None, features_only=True)[FT_KEY]