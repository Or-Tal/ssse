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
        self.window_size = loss_cfg.window_size

    def f(self, a, b):
        return 1 + F.cosine_similarity(a, b, -1)
    
    def match_vad_to_windows(self, vad_mask, device):
        # print(f"vad: {vad_mask.shape}")
        num_windows = vad_mask.shape[-1] // self.window_size
        n = num_windows * self.window_size
        mask = vad_mask[..., :n].float()
        # print(f"mask: {vad_mask.shape}")
        mask_windows = [torch.sum(vad_mask[..., i:i+self.window_size], dim=-1) for i in range(0, n+1, self.window_size)]
        # print(f"mask_windows -> len: {len(mask_windows)}, inner: {mask_windows[0].shape}")
        mask = torch.stack(mask_windows, dim=-1)
        # print(f"concatenated mask: {mask.shape}")
        mask = mask > (self.window_size/2)
        return mask.to(device)

    def get_divisors(self, w_c, w_n, denoms=None):
        # init list of negative dot products, each of shape: (B, T, Ft)
        # after the similarity function (f): (B, T)
        # note: denoms has a temporal size of T - 1 as it is the result of the denominator's dot products
        # print(f"denoms: {'none' if denoms is None else denoms.shape}")
        # print(f"w_n: {w_n.shape}")
        # print(f"w_c: {w_n.shape}")
        neg_dots = [denoms + EPS, torch.exp(self.f(w_c, w_n))[:, :-1]] if denoms is not None else [torch.exp(self.f(w_c, w_n))]
        for _ in range(NEG_SIZE - 1):
            perm = torch.randperm(w_n.shape[1])
            neg_dots.append(torch.exp(self.f(w_c, w_n[:, perm]))[:, :-1] if denoms is not None else torch.exp(self.f(w_c, w_n[:, perm])))
        
        # stack all denominators (calculated similarities permutations w.r.t w_c)
        neg_dots = torch.stack(neg_dots, dim=-1) # shape: (B, T, NEG_SIZE + 1 (or NEG_SIZE if denoms is None))

        # sum 
        neg_dots = torch.sum(neg_dots, dim=-1)

        return 1 / neg_dots
    

    def contrastive_loss(self, w_c, w_n, vad_mask, device):
        # permute for simplicity
        w_c = w_c.permute((0, 2, 1))  # batch x T x Ft
        w_n = w_n.permute((0, 2, 1))  # batch x T x Ft

        # match vad to windows
        shrinked_vad = self.match_vad_to_windows(vad_mask, device)

        shrinked_vad = shrinked_vad.to(float)
        # TODO: is + eps really necessary?
        # shrinked_vad = shrinked_vad + EPS

        # TODO: remove limitation of length to vad (n < 0) and uncomment line 34
        n = w_c.shape[1] - shrinked_vad.shape[-1]
        # n = max(w_c.shape[1] - vad_mask.shape[-1], 0)
        if n > 0:
            shrinked_vad = torch.cat([shrinked_vad, torch.zeros((vad_mask.shape[0], n), device=device)], dim=-1)
        elif n < 0:
            shrinked_vad = shrinked_vad[..., :w_c.shape[1]]
        w_c = w_c * shrinked_vad.unsqueeze(2).expand_as(w_c)
        w_ci, w_cip1 = w_c[:, :-1, :], w_c[:, 1:, :]

        # calculate denominators
        denominators = torch.exp(self.f(w_ci, w_cip1)) * shrinked_vad[..., :-1] # shape: (B, T)

        # calculate divisor factors
        divisors = self.get_divisors(w_c, w_n, denoms=denominators) # shape: (B, T)

        N = torch.sum(shrinked_vad, dim=-1)

        # calculate loss terms
        if divisors.shape != denominators.shape:
            divisors = divisors.unsqueeze(-1).expand_as(denominators)
        terms = torch.sum(- torch.log(denominators * divisors), dim=-1)
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

        contrastive_loss = self.contrastive_loss(w_c, w_n, self.match_vad_to_windows(vad_mask, device), device) if self.include_contrastive else 0
        reg_loss = self.regularization_loss(vad_mask, z_hat, noisy_sigs) if self.include_regularization else 0

        return [self.reconstruction_factor * reconstruction_loss, self.contrastive_factor * contrastive_loss, self.noise_regularization_factor * reg_loss]



class SupSELoss(SELoss):

    def forward(self, outputs, noisy_sigs, clean_sigs, vad_mask):
        _, _, y_hat, z_hat, w_c, w_n = outputs
        device = f"{f'cuda' if w_c.is_cuda else 'cpu'}"
        y_hat, z_hat, w_c, w_n, noisy_sigs, clean_sigs, vad_mask = (y_hat.to(device), z_hat.to(device), w_c.to(device), w_n.to(device), 
                                                                    noisy_sigs.to(device), clean_sigs.to(device), vad_mask.to(device))
        if noisy_sigs.shape[-1] > y_hat.shape[-1]:
            noisy_sigs = noisy_sigs[..., :y_hat.shape[-1]]
            clean_sigs = clean_sigs[..., :y_hat.shape[-1]]
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

        contrastive_loss = self.contrastive_loss(w_c, w_n, self.match_vad_to_windows(vad_mask, device), device) if self.include_contrastive else 0
        reg_loss = self.regularization_loss(vad_mask, z_hat, noisy_sigs) if self.include_regularization else 0

        return [self.reconstruction_factor * reconstruction_loss, self.contrastive_factor * contrastive_loss, self.noise_regularization_factor * reg_loss]

