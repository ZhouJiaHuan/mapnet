"""
implementation of PoseNet and MapNet networks
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init

from ..registry import MODELS


@MODELS.register_module(name='PoseNet')
class PoseNet(nn.Module):
    '''posenet with `resnet` for feature extraction
    '''
    def __init__(self, backbone, droprate=0.5, pretrained=True,
                 feat_dim=2048):
        super(PoseNet, self).__init__()
        self.droprate = droprate

        # replace the last FC layer in feature extractor
        assert hasattr(backbone, 'model')
        self.backbone = backbone.model
        self.backbone.avgpool = nn.AdaptiveAvgPool2d(1)
        fe_out_planes = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(fe_out_planes, feat_dim)

        self.fc_xyz = nn.Linear(feat_dim, 3)
        self.fc_wpqr = nn.Linear(feat_dim, 3)

        # initialize
        self._init_weights(pretrained)

    def _init_weights(self, pretrained=True):
        if pretrained:
            init_modules = [self.backbone.fc, self.fc_xyz, self.fc_wpqr]
        else:
            init_modules = self.modules()

        for m in init_modules:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        x = self.backbone(x)
        x = F.relu(x)
        if self.droprate > 0:
            x = F.dropout(x, p=self.droprate, training=self.training)
        xyz = self.fc_xyz(x)
        wpqr = self.fc_wpqr(x)
        return torch.cat((xyz, wpqr), 1)


@MODELS.register_module(name='MapNet')
class MapNet(PoseNet):
    ''' mapnet class derived from PoseNet
    for the relative loss function in mapnet, the input dim is
    different from that in PoseNet. see `MapNetCriterion` for
    more details.
    '''
    def __init__(self, *args, **kwargs):
        super(MapNet, self).__init__(*args, **kwargs)

    def forward(self, x):
        s = x.size()
        x = x.view(-1, *s[2:])
        poses = super().forward(x)
        poses = poses.view(s[0], s[1], -1)
        return poses
