# Author: Or Tal.
import omegaconf
import torch.nn as nn
import torch.nn.functional as F
import torch

from .multi_res_stft_loss import MultiResolutionSTFTLoss

EPS = 1e-10
NEG_SIZE = 10

class SELoss(nn.Module):

    def __init__(self,loss_cfg: omegaconf.DictConfig, device='cpu'):
        super().__init__()
        device = device if torch.cuda.is_available() else 'cpu'
        self.m_stft_loss = MultiResolutionSTFTLoss(factor_sc=loss_cfg.stft_sc_factor, 
        factor_mag=loss_cfg.stft_mag_factor).to(device)
        self.just_reconstruction = loss_cfg.just_reconstruction
        self.reconstruction_factor = loss_cfg.reconstruction_factor
        self.contrastive_factor = loss_cfg.contrastive_factor
        self.noise_regularization_factor = loss_cfg.noise_regularization_factor
        self.mse = nn.MSELoss(reduction='none')
        self.include_regularization = loss_cfg.include_regularization
        self.include_contrastive = loss_cfg.include_contrastive

    def f(self, a, b):
        return 1 + F.cosine_similarity(a, b, -1)

    def get_divisors(self, w_c, w_n, vad_mask):
        neg_dots = torch.sum(torch.exp(self.f(w_c, w_n)) * vad_mask)
        for _ in range(NEG_SIZE - 1):
            perm = torch.randperm(w_n.shape[1])
            neg_dots = neg_dots + torch.sum(torch.exp(self.f(w_c, w_n[:, perm])) * vad_mask[:, perm])
        neg_dots = neg_dots / NEG_SIZE
        return 1 / (neg_dots + EPS)

    def contrastive_loss(self, w_c, w_n, vad_mask, device):
        # permute for simplicity
        w_c = w_c.permute((0, 2, 1))  # batch x T x Ft
        w_n = w_n.permute((0, 2, 1))  # batch x T x Ft

        vad_mask = vad_mask.to(float)
        # TODO: is + eps really necessary?
        vad_mask = vad_mask + EPS

        # TODO: remove limitation of length to vad (n < 0) and uncomment line 34
        n = w_c.shape[1] - vad_mask.shape[-1]
        # n = max(w_c.shape[1] - vad_mask.shape[-1], 0)
        if n > 0:
            vad_mask = torch.cat([vad_mask, torch.zeros((vad_mask.shape[0], n), device=device)], dim=-1)
        elif n < 0:
            vad_mask = vad_mask[..., :w_c.shape[1]]
        w_c = w_c * vad_mask.unsqueeze(2).expand_as(w_c)
        w_ci, w_cip1 = w_c[:, :-1, :], w_c[:, 1:, :]

        # calculate divisor factors
        divisors = self.get_divisors(w_c, w_n, vad_mask)

        # calculate denominators
        denominators = torch.exp(self.f(w_ci, w_cip1)) * vad_mask[..., :-1]
        N = torch.count_nonzero(vad_mask, dim=-1)

        # calculate loss terms
        terms = torch.sum(- torch.log(EPS + denominators * divisors.unsqueeze(-1).expand_as(denominators)) * vad_mask[..., :-1], dim=-1)
        terms = terms / N

        return torch.mean(terms)

    def stretch_vad_mask_over_input_length(self, vad_mask, noisy):
        factor = noisy.shape[-1] // vad_mask.shape[-1]
        vad_mask = vad_mask.unsqueeze(-1)
        vad_mask = vad_mask.expand(-1, -1, factor)
        return vad_mask.reshape(vad_mask.shape[0], -1)


    def validate_lengths(self, stretched_nad, z_hat, noisy):
        if z_hat.shape[-1] > stretched_nad.shape[-1]:
            z_hat = z_hat[..., :stretched_nad.shape[-1]]
        elif stretched_nad.shape[-1] > z_hat.shape[-1]:
            stretched_nad = stretched_nad[..., :z_hat.shape[-1]]
        if noisy.shape[-1] > stretched_nad.shape[-1]:
            noisy = noisy[..., :stretched_nad.shape[-1]]

        return stretched_nad, z_hat, noisy

    def regularization_loss(self, vad_mask, z_hat, noisy):

        # stretch vad_maps
        stretched_vad = self.stretch_vad_mask_over_input_length(vad_mask, noisy).unsqueeze(1)

        stretched_vad, z_hat, noisy = self.validate_lengths(stretched_vad, z_hat, noisy)

        stretched_vad = stretched_vad.bool()
        noises_from_z = torch.masked_select(z_hat, ~stretched_vad).flatten()
        noises_from_noisy = torch.masked_select(noisy, ~stretched_vad).flatten()
        return F.mse_loss(noises_from_z, noises_from_noisy)

    def forward(self, outputs, noisy_sigs, vad_mask):
        l_c, l_n, y_hat, z_hat, w_c, w_n = outputs
        device = f"{f'cuda' if w_c.is_cuda else 'cpu'}"
        if noisy_sigs.shape[-1] > y_hat.shape[-1]:
            noisy_sigs = noisy_sigs[..., :y_hat.shape[-1]]
        elif noisy_sigs.shape[-1] < y_hat.shape[-1]:
            y_hat = y_hat[..., :noisy_sigs.shape[-1]]
            z_hat = z_hat[..., :noisy_sigs.shape[-1]]

        est_noisy = y_hat + z_hat
        fc, mag = self.m_stft_loss(est_noisy.squeeze(1), noisy_sigs.squeeze(1))
        if torch.isnan(est_noisy).any():
            print("passed nan value from ae")
        reconstruction_loss = F.l1_loss(est_noisy, noisy_sigs).to(device) + fc + mag
        if self.just_reconstruction:
            return reconstruction_loss

        contrastive_loss = self.contrastive_loss(w_c, w_n, vad_mask, device) if self.include_contrastive else 0
        reg_loss = self.regularization_loss(vad_mask, z_hat, noisy_sigs) if self.include_regularization else 0

        return self.reconstruction_factor * reconstruction_loss + self.contrastive_factor * contrastive_loss + \
               self.noise_regularization_factor * reg_loss



class SupSELoss(SELoss):

    def forward(self, outputs, noisy_sigs, clean_sigs, vad_mask):
        _, _, y_hat, z_hat, w_c, w_n = outputs
        device = f"{f'cuda' if w_c.is_cuda else 'cpu'}"
        if noisy_sigs.shape[-1] > y_hat.shape[-1]:
            noisy_sigs = noisy_sigs[..., :y_hat.shape[-1]]
        elif noisy_sigs.shape[-1] < y_hat.shape[-1]:
            y_hat = y_hat[..., :noisy_sigs.shape[-1]]
            z_hat = z_hat[..., :noisy_sigs.shape[-1]]

        est_noisy = y_hat + z_hat
        fc, mag = self.m_stft_loss(est_noisy.squeeze(1), noisy_sigs.squeeze(1))
        if torch.isnan(est_noisy).any():
            print("passed nan value from ae")
        fc_c, mag_c = self.m_stft_loss(y_hat.squeeze(1), clean_sigs.squeeze(1))
        reconstruction_loss = F.l1_loss(est_noisy, noisy_sigs).to(device) + fc + mag + fc_c + mag_c
        if self.just_reconstruction:
            return reconstruction_loss

        contrastive_loss = self.contrastive_loss(w_c, w_n, vad_mask, device) if self.include_contrastive else 0
        reg_loss = self.regularization_loss(vad_mask, z_hat, noisy_sigs) if self.include_regularization else 0

        return self.reconstruction_factor * reconstruction_loss + self.contrastive_factor * contrastive_loss + \
               self.noise_regularization_factor * reg_loss

