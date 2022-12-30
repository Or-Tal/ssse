import torch
from .visqol import ViSQOL

# from .. import losses
# from ..utils.utils import dict_from_config
# import omegaconf
# import torch

from pesq import pesq
from pystoi import stoi
import numpy as np


# def get_visqol(cfg: omegaconf.DictConfig):
#     """Instantiate ViSQOL from config.
#     """
#     kwargs = dict_from_config(cfg)
#     return ViSQOL(**kwargs)

# def evaluate_audio_reconstruction(y_pred: torch.Tensor, y: torch.Tensor, cfg: omegaconf.DictConfig) -> dict:
#     """Evaluate audio reconstruction, returning the metrics specified in the configuration.

#     Args:
#         y_pred (torch.Tensor): Reconstructed audio.
#         y (torch.Tensor): Reference audio.

#     Returns:
#         dict: Dictionary of metrics.
#     """
#     metrics = {}
#     if cfg.evaluate.metrics.visqol:
#         visqol = get_visqol(cfg.metrics.visqol)
#         metrics['visqol'] = visqol(y_pred, y, cfg.sample_rate)
#     sisnr = losses.get_loss('sisnr', cfg)
#     metrics['sisnr'] = sisnr(y_pred, y)
#     return metrics


def get_pesq(ref_sig, out_sig, sr):
    """Calculate PESQ.
    Args:
        ref_sig: numpy.ndarray, [B, T]
        out_sig: numpy.ndarray, [B, T]
    Returns:
        PESQ
    """
    if not isinstance(out_sig, np.ndarray):
        out_sig = out_sig.cpu().numpy()
    pesq_val = 0
    B = ref_sig.shape[0]
    for i in range(len(ref_sig)):
        pesq_val += pesq(sr, ref_sig[i], out_sig[i], 'wb')
    return pesq_val / B


def get_stoi(ref_sig, out_sig, sr):
    """Calculate STOI.
    Args:
        ref_sig: numpy.ndarray, [B, T]
        out_sig: numpy.ndarray, [B, T]
    Returns:
        STOI
    """
    stoi_val = 0
    B = ref_sig.shape[0]
    for i in range(len(ref_sig)):
        stoi_val += stoi(ref_sig[i], out_sig[i], sr, extended=False)
    return stoi_val / B


def get_snr(signal, noise):
    return 10 * torch.log10((signal**2)/(noise**2 + 1e-10)).mean()

