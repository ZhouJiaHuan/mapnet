import torch
from torchvision import transforms
from mapnet.datasets.compose import Compose
from ..registry import PIPELINES


def build_transforms(trans_cfgs):
    data_transform = Compose(trans_cfgs)
    target_transform = transforms.Lambda(lambda x: torch.from_numpy(x).float())
    return data_transform, target_transform


@PIPELINES.register_module(name='ToTensor')
class ToTensor(object):
    def __call__(self, pic):
        return transforms.ToTensor()(pic=pic)


@PIPELINES.register_module(name='Resize')
class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        return transforms.Resize(self.size)(img=img)


@PIPELINES.register_module(name='Normalize')
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return transforms.Normalize(self.mean, self.std)(tensor=tensor)


@PIPELINES.register_module(name='ColorJitter')
class ColorJitter(object):
    def __init__(self, brightness, contrast, saturation, hue):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, img):
        return transforms.ColorJitter(self.brightness,
                                      self.contrast,
                                      self.saturation,
                                      self.hue)(img=img)


@PIPELINES.register_module(name='RandomCrop')
class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        return transforms.RandomCrop(self.size)(img=img)


@PIPELINES.register_module(name='CenterCrop')
class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        return transforms.CenterCrop(self.size)(img=img)
