import torch.nn as nn
from torchvision import models

from ..registry import BACKBONES


@BACKBONES.register_module(name='ResNet34')
class ResNet34(nn.Module):

    def __init__(self, pretrained):
        super(ResNet34, self).__init__()
        self.model = models.resnet34(pretrained=pretrained)

    def forward(self, x):
        return self.model(x)
