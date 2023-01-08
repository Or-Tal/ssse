from typing import List, Union
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


class DualAEJointEncoder(nn.Module):

    def __init__(self, encoder: nn.Module, decoder: nn.Module, ft_encoder: nn.Module, 
                 quantizer: Union[None, nn.Module], include_skips_in_fw = True):

        super().__init__()
        self.decoder = decoder
        self.ft_enc = ft_encoder
        self.encoder = encoder
        self.quantizer = quantizer

    def infer(self, mix):
        with torch.no_grad():
            _, _, y_hat, z_hat, _, _ = self.forward(mix)
        return y_hat, z_hat


    def valid_length(self, length):
        return self.encoder.valid_length(length)
    
    @staticmethod
    def split_to_noisy_clean(ls, skips):
        # ls: {Batch, Ft, T}
        # skips: [{Batch, Ft, T}]
        
        def split_single_val(val):
            b, _, t = val.shape
            val = val.reshape((b, -1, 2, t))
            return val[..., 0, :], val[..., 1, :]
        
        l_c, l_n = split_single_val(ls)
        skips_c, skips_n = [], []
        for s in skips:
            s_c, s_n = split_single_val(s)
            skips_c.append(s_c)
            skips_n.append(s_n)
        
        return l_c, l_n, skips_c, skips_n


    def forward(self, mix, eval=False):
        ls, skips, std, length = self.encoder(mix, include_skips=True, include_std_len=True)
        ls, l_n, skips_c, skips_n = self.split_to_noisy_clean(ls, skips)
        
        if self.quantizer is not None:
            ls = self.quantizer(ls)
        y_hat = self.decoder(ls, [s for s in skips_c], std, length)
        if eval:
            return y_hat
        if self.quantizer is not None:
            l_n = self.quantizer(l_n)
        z_hat = self.decoder(l_n, [s for s in skips_n], std, length)

        return ls, l_n, y_hat, z_hat, self.ft_enc(ls), self.ft_enc(l_n)
