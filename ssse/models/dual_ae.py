from typing import List
import torch.nn as nn
import torch

class DualAE(nn.Module):

    def __init__(self, encoders: List[nn.Module], decoder: nn.Module, ft_encoder: nn.Module, include_skips_in_fw = True):
        super().__init__()
        self.encoder_c = encoders[0]
        self.encoder_n = encoders[1]
        self.decoder = decoder
        self.ft_enc = ft_encoder


    def infer(self, mix):
        with torch.no_grad():
            _, _, y_hat, z_hat, _, _ = self.forward(mix)
        return y_hat, z_hat


    def valid_length(self, length):
        return self.encoder.valid_length(length)

    def forward(self, mix, eval=False):
        l_c, skips_c, std_c, length_c = self.encoder_c(mix, include_skips=True, include_std_len=True)
        y_hat = self.decoder(l_c, [s for s in skips_c], std_c, length_c)
        if eval:
            return y_hat

        l_n, skips_n, std_n, length_n = self.encoder_n(mix, include_skips=True, include_std_len=True)
        z_hat = self.decoder(l_n, [s for s in skips_n], std_n, length_n)

        return l_c, l_n, y_hat, z_hat, self.ft_enc(l_c), self.ft_enc(l_n)
