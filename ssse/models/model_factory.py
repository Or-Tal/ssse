# Author: Or Tal
import omegaconf
import torch. nn as nn
from .demucs_ae import DemucsConfig, DemucsDecoder, DemucsDoubleAEwJointEncoder, DemucsEncoder
from magma.utils.utils import dict_from_config
from .dual_ae import DualAE
from .feature_encoder import FeatureEncoderBLSTM
from .hifi_gan_generator import HifiGanGenerator, HifiGeneratorConfig
from .wav2vec2_encoder import Wav2vec2Encoder

_supported_modules = {
    'demucs_encoder': DemucsEncoder,
    'demucs_decoder': DemucsDecoder,
    'wav2vec_encoder': Wav2vec2Encoder,
    'hifi_gan_generator': HifiGanGenerator,
    'blstm_feature': FeatureEncoderBLSTM,
    'demucs_joint_ae': DemucsDoubleAEwJointEncoder,
}

_supported_configs = {
    'demucs': DemucsConfig,
    'hifi_gan_generator': HifiGeneratorConfig
}

def create_config(cfg: omegaconf.DictConfig, cfg_name):
    if "demucs" in cfg_name.lower():
        return _supported_configs["demucs"](**cfg.demucs)
    elif cfg_name.lower() in _supported_configs.keys():
        return _supported_configs[cfg_name.lower()](**getattr(cfg, cfg_name.lower()))
    else:
        return dict_from_config(cfg)


def model_factory(cfg: omegaconf.DictConfig, model_class_name: str) -> nn.Module:
    if model_class_name.lower() == "se_dual_ae":
        encoders = [model_factory(cfg, cfg.model.encoder_model), model_factory(cfg, cfg.model.encoder_model)]
        decoder = model_factory(cfg, cfg.model.decoder_model)
        feature_model = model_factory(cfg, cfg.model.feature_model)
        return DualAE(encoders, decoder, feature_model, cfg.model.include_skips_in_fw_pass)
    elif model_class_name.lower() in _supported_modules.keys():
        return _supported_modules[model_class_name.lower()](create_config(cfg.model, model_class_name.lower()))
    else:
        raise ValueError(f"invalid model class: {cfg.model.model_class_name}")