from magma.magma.quantization.vq import ResidualVectorQuantizer
import torch.nn as nn

class RVQ(nn.Module):

    def __init__(self, quantizer_args, sample_rate=16000):
        self.sample_rate = sample_rate
        self.quantizer = ResidualVectorQuantizer(**quantizer_args)
    
    def forward(self, x):
        return self.quantizer.forward(x, self.sample_rate)
