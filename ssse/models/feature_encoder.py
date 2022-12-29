import torch.nn as nn
import torch

class FeatureEncoderBLSTM(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.hidden_size = args['feature_encoder']['hidden_size']
        self.window_size = args['feature_encoder']['window_size']
        self.ft_extractor_fw = nn.LSTM(input_size=768, hidden_size=self.hidden_size, num_layers=2, batch_first=True,
                                       bidirectional=False)
        self.ft_extractor_bw = nn.LSTM(input_size=768, hidden_size=self.hidden_size, num_layers=2, batch_first=True,
                                       bidirectional=False)
        self.final_proj = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.tanh = nn.Tanh()

    def forward_through_batch_of_windows(self, forward_batch, backward_batch):
        """
        batches_shape: {B * num_windows, win_size, Ft}
        """
        windows_fw, _ = self.ft_extractor_fw(forward_batch)
        windows_bw, _ = self.ft_extractor_bw(backward_batch)
        windows_fw = windows_fw.permute(0, 2, 1)
        windows_bw = windows_bw.permute(0, 2, 1)
        windows = torch.concatenate([windows_fw[..., -1], windows_bw[..., -1]], dim=-1)  # many to one
        return self.tanh(self.final_proj(windows))

    def forward(self, latent_vectors):
        """
        input: {Batch, Ft, T} | given self.window_size
        output: (windows : {Batch * (T//self.window_size), Ft, self.window_size},
                 residual_window: {Batch, Ft, T - (T//self.window_size) * self.window_size})
        """
        B, Ft, T = latent_vectors.shape
        num_windows = T // self.window_size
        n = self.window_size * num_windows

        windows = self.forward_through_batch_of_windows(
            latent_vectors[..., :n].reshape(B * num_windows, Ft, self.window_size).permute(0, 2, 1),
                                            torch.flip(latent_vectors[..., :n], dims=(-1,)).reshape(B * num_windows, Ft, self.window_size).permute(0, 2, 1))
        if T > n:  # sequence has a residue
            res_windows = self.forward_through_batch_of_windows(latent_vectors[..., n:].permute(0, 2, 1),
                                                                torch.flip(latent_vectors[..., n:], dims=(-1,)).permute(0, 2, 1))
            windows = torch.concatenate([windows, res_windows], dim=0)
        return windows.reshape(B, self.hidden_size, -1)