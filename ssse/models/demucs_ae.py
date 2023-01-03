from typing import List
import omegaconf
import torch.nn as nn
from dataclasses import dataclass
import torch
import math
from torch.nn import functional as F
from .feature_encoder import FeatureEncoderBLSTM



def sinc(t):
    """sinc.
    :param t: the input tensor
    """
    return torch.where(t == 0, torch.tensor(1., device=t.device, dtype=t.dtype), torch.sin(t) / t)


def kernel_upsample2(zeros=56):
    """kernel_upsample2.
    """
    win = torch.hann_window(4 * zeros + 1, periodic=False)
    winodd = win[1::2]
    t = torch.linspace(-zeros + 0.5, zeros - 0.5, 2 * zeros)
    t *= math.pi
    kernel = (sinc(t) * winodd).view(1, 1, -1)
    return kernel


def upsample2(x, zeros=56):
    """
    Upsampling the input by 2 using sinc interpolation.
    Smith, Julius, and Phil Gossett. "A flexible sampling-rate conversion method."
    ICASSP'84. IEEE International Conference on Acoustics, Speech, and Signal Processing.
    Vol. 9. IEEE, 1984.
    """
    *other, time = x.shape
    kernel = kernel_upsample2(zeros).to(x)
    out = F.conv1d(x.view(-1, 1, time), kernel, padding=zeros)[..., 1:].view(*other, time)
    y = torch.stack([x, out], dim=-1)
    return y.view(*other, -1)


def kernel_downsample2(zeros=56):
    """kernel_downsample2.
    """
    win = torch.hann_window(4 * zeros + 1, periodic=False)
    winodd = win[1::2]
    t = torch.linspace(-zeros + 0.5, zeros - 0.5, 2 * zeros)
    t.mul_(math.pi)
    kernel = (sinc(t) * winodd).view(1, 1, -1)
    return kernel


def downsample2(x, zeros=56):
    """
    Downsampling the input by 2 using sinc interpolation.
    Smith, Julius, and Phil Gossett. "A flexible sampling-rate conversion method."
    ICASSP'84. IEEE International Conference on Acoustics, Speech, and Signal Processing.
    Vol. 9. IEEE, 1984.
    """
    if x.shape[-1] % 2 != 0:
        x = F.pad(x, (0, 1))
    xeven = x[..., ::2]
    xodd = x[..., 1::2]
    *other, time = xodd.shape
    kernel = kernel_downsample2(zeros).to(x)
    out = xeven + F.conv1d(xodd.view(-1, 1, time), kernel, padding=zeros)[..., :-1].view(
        *other, time)
    return out.view(*other, -1).mul(0.5)


@dataclass
class DemucsConfig:
    chin: int = 1
    chout: int = 1
    hidden: int = 48
    depth: int = 5
    kernel_size: int = 8
    stride: int = 4
    causal: bool = True
    resample: int = 4
    growth: int = 2
    max_hidden: int = 10_000
    normalize: bool = True
    glu: bool = True
    rescale: float = 0.1
    floor: float = 1e-3


class BLSTM(nn.Module):
    def __init__(self, dim, layers=2, bi=True):
        super().__init__()
        klass = nn.LSTM
        self.lstm = klass(bidirectional=bi, num_layers=layers, hidden_size=dim, input_size=dim)
        self.linear = None
        if bi:
            self.linear = nn.Linear(2 * dim, dim)

    def forward(self, x, hidden=None):
        x, hidden = self.lstm(x, hidden)
        if self.linear:
            x = self.linear(x)
        return x, hidden


def rescale_conv(conv, reference):
    std = conv.weight.std().detach()
    scale = (std / reference) ** 0.5
    conv.weight.data /= scale
    if conv.bias is not None:
        conv.bias.data /= scale


def rescale_module(module, reference):
    for sub in module.modules():
        if isinstance(sub, (nn.Conv1d, nn.ConvTranspose1d)):
            rescale_conv(sub, reference)


class DemucsEncoder(nn.Module):
    """
    Demucs speech enhancement model.
    Args:
        - chin (int): number of input channels.
        - chout (int): number of output channels.
        - hidden (int): number of initial hidden channels.
        - depth (int): number of layers.
        - kernel_size (int): kernel size for each layer.
        - stride (int): stride for each layer.
        - causal (bool): if false, uses BiLSTM instead of LSTM.
        - resample (int): amount of resampling to apply to the input/output.
            Can be one of 1, 2 or 4.
        - growth (float): number of channels is multiplied by this for every layer.
        - max_hidden (int): maximum number of channels. Can be useful to
            control the size/speed of the model.
        - normalize (bool): if true, normalize the input.
        - glu (bool): if true uses GLU instead of ReLU in 1x1 convolutions.
        - rescale (float): controls custom weight initialization.
            See https://arxiv.org/abs/1911.13254.
        - floor (float): stability flooring when normalizing.
    """

    def __init__(self, args: DemucsConfig):

        chin = args.chin
        chout = args.chout
        hidden = args.hidden
        depth = args.depth
        kernel_size = args.kernel_size
        stride = args.stride
        causal = args.causal
        resample = args.resample
        growth = args.growth
        max_hidden = args.max_hidden
        normalize = args.normalize
        glu = args.glu
        rescale = args.rescale
        floor = args.floor
        self.args = args

        super().__init__()
        if resample not in [1, 2, 4]:
            raise ValueError(f"Resample should be 1, 2 or 4. value: {resample}")

        self.chin = chin
        self.chout = chout
        self.hidden = hidden
        self.depth = depth
        self.kernel_size = kernel_size
        self.stride = stride
        self.causal = causal
        self.floor = floor
        self.resample = resample
        self.normalize = normalize

        self.encoder = nn.ModuleList()
        activation = nn.GLU(1) if glu else nn.ReLU()
        ch_scale = 2 if glu else 1

        for index in range(depth):
            encode = []
            encode += [
                nn.Conv1d(chin, hidden, kernel_size, stride),
                nn.ReLU(),
                nn.Conv1d(hidden, hidden * ch_scale, 1), activation,
            ]
            self.encoder.append(nn.Sequential(*encode))
            chout = hidden
            chin = hidden
            hidden = min(int(growth * hidden), max_hidden)

        self.lstm = BLSTM(chin, bi=not causal)
        if rescale:
            rescale_module(self, reference=rescale)

    @property
    def total_stride(self):
        return self.stride ** self.depth // self.resample

    def forward(self, mix, include_skips=True, include_std_len=True):
        if mix.dim() == 2:
            mix = mix.unsqueeze(1)

        if self.normalize:
            mono = mix.mean(dim=1, keepdim=True)
            std = mono.std(dim=-1, keepdim=True)
            mix = mix / (self.floor + std)
        else:
            std = 1
        length = mix.shape[-1]
        x = mix
        # x = F.pad(x, (0, self.valid_length(length) - length))
        if self.resample == 2:
            x = upsample2(x)
        elif self.resample == 4:
            x = upsample2(x)
            x = upsample2(x)
        skips = []
        for encode in self.encoder:
            x = encode(x)
            if include_skips:
                skips.append(x)
        x = x.permute(2, 0, 1)
        x, _ = self.lstm(x)
        x = x.permute(1, 2, 0)

        if include_std_len:
            if include_skips:
                ret = (x, skips, std, length)
            else:
                ret = (x, std, length)
        elif include_skips:
            ret = (x, skips)
        else:
            ret = x

        return ret


class DemucsDecoder(nn.Module):
    """
    Demucs speech enhancement model.
    Args:
        - chin (int): number of input channels.
        - chout (int): number of output channels.
        - hidden (int): number of initial hidden channels.
        - depth (int): number of layers.
        - kernel_size (int): kernel size for each layer.
        - stride (int): stride for each layer.
        - causal (bool): if false, uses BiLSTM instead of LSTM.
        - resample (int): amount of resampling to apply to the input/output.
            Can be one of 1, 2 or 4.
        - growth (float): number of channels is multiplied by this for every layer.
        - max_hidden (int): maximum number of channels. Can be useful to
            control the size/speed of the model.
        - normalize (bool): if true, normalize the input.
        - glu (bool): if true uses GLU instead of ReLU in 1x1 convolutions.
        - rescale (float): controls custom weight initialization.
            See https://arxiv.org/abs/1911.13254.
        - floor (float): stability flooring when normalizing.
    """
    def __init__(self, args: DemucsConfig):

        chin = args.chin
        chout = args.chout
        hidden = args.hidden
        depth = args.depth
        kernel_size = args.kernel_size
        stride = args.stride
        causal = args.causal
        resample = args.resample
        growth = args.growth
        max_hidden = args.max_hidden
        normalize = args.normalize
        glu = args.glu
        rescale = args.rescale
        floor = args.floor
        self.args = args

        super().__init__()
        if resample not in [1, 2, 4]:
            raise ValueError("Resample should be 1, 2 or 4.")

        self.chin = chin
        self.chout = chout
        self.hidden = hidden
        self.depth = depth
        self.kernel_size = kernel_size
        self.stride = stride
        self.causal = causal
        self.floor = floor
        self.resample = resample
        self.normalize = normalize

        self.decoder = nn.ModuleList()
        activation = nn.GLU(1) if glu else nn.ReLU()
        ch_scale = 2 if glu else 1

        for index in range(depth):

            decode = []
            decode += [
                nn.Conv1d(hidden, ch_scale * hidden, 1), activation,
                nn.ConvTranspose1d(hidden, chout, kernel_size, stride),
            ]
            if index > 0:
                decode.append(nn.ReLU())
            self.decoder.insert(0, nn.Sequential(*decode))
            chout = hidden
            chin = hidden
            hidden = min(int(growth * hidden), max_hidden)

    @property
    def total_stride(self):
        return self.stride ** self.depth // self.resample

    def forward(self, x, skips, std, length):

        for i, decode in enumerate(self.decoder):
            skip = skips[-(i+1)]
            x = x + skip[..., :x.shape[-1]]
            x = decode(x)
        if self.resample == 2:
            x = downsample2(x)
        elif self.resample == 4:
            x = downsample2(x)
            x = downsample2(x)

        x = x[..., :length]

        return std * x

class DemucsJointEncoder(DemucsEncoder):
    
    def __init__(self, args: DemucsConfig):
        args.hidden = 2 * args.hidden
        super().__init__(args)


class DemucsDoubleAE(nn.Module):

    def __init__(self, args: DemucsConfig, mutual_enc_layers=0):
        super().__init__()
        self.encoder_c = DemucsEncoder(args)
        self.encoder_n = DemucsEncoder(args)
        self.mutual_enc_layers = mutual_enc_layers
        if mutual_enc_layers > 0:
            self.mutual_enc = None
        
        self.decoder = DemucsDecoder(args)
        self.ft_enc = FeatureEncoderBLSTM(args)


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
