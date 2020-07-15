from mmcv.utils import build_from_cfg
from .registry import MODELS, DATASETS, BACKBONES, LOSSES, OPTIMIZER


def build_model(model_cfg, default_args=None):
    return build_from_cfg(model_cfg, MODELS, default_args)


def build_dataset(data_cfg, default_args=None):
    return build_from_cfg(data_cfg, DATASETS, default_args)


def build_backbone(backbone_cfg, default_args=None):
    return build_from_cfg(backbone_cfg, BACKBONES, default_args)


def build_loss(loss_cfg, default_args=None):
    return build_from_cfg(loss_cfg, LOSSES, default_args)


def build_optimizer(optim_cfg, default_args=None):
    return build_from_cfg(optim_cfg, OPTIMIZER, default_args)
