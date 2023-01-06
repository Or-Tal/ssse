from .se_loss import *

class NaiveSELoss(SELoss):

    def single_sample_contrastive_loss(self, w_c, w_n, vad_mask):
        # w_c.shape = w_n.shape = (T, Ft)
        # vad_mask.shape = (T,)
        # print(f"vad: {vad_mask.shape}, w_c: {w_c.shape}")
        denominators = []
        divisors = []
        for i in range(len(w_c)-1):
            if vad_mask[i] and vad_mask[i+1]:
                denom = torch.exp(self.f(w_c[i], w_c[i+1]))
                denominators.append(denom)
                tmp = w_n[torch.randperm(w_n.shape[0])][:NEG_SIZE]
                divisor = denom
                for w in tmp:
                    divisor = divisor + torch.exp(self.f(w_c[i], w))
                divisors.append(divisor)
        denominators = torch.stack(denominators, dim=0)
        divisors = torch.stack(divisors, dim=0)
        return - torch.mean(torch.log(denominators / (divisors + EPS)))


    def contrastive_loss(self, w_c, w_n, vad_mask, device):
        # permute for simplicity
        w_c = w_c.permute((0, 2, 1))  # batch x T x Ft
        w_n = w_n.permute((0, 2, 1))  # batch x T x Ft
        vad_mask_windows = self.match_vad_to_windows(vad_mask, device)

        # # print sizes for debugging
        # print("---- sizes ----")
        # print(f"vad: {vad_mask_windows.shape}")
        # print(f"w_c: {w_c.shape}")
        # print(f"w_n: {w_n.shape}")
        # print("----------------")

        results = torch.tensor([
            self.single_sample_contrastive_loss(w_c_i, w_n_i, vad_mask_i) for w_c_i, w_n_i, vad_mask_i in zip(w_c, w_n, vad_mask_windows)
        ])

        return torch.mean(results)