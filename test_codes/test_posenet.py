import torch

import sys
sys.path.append('.')
from mapnet.models import PoseNet, MapNet, ResNet34


resnet34 = ResNet34(pretrained=True)

posenet = PoseNet(resnet34)
mapnet = MapNet(resnet34)

x1 = torch.randn([20, 3, 224, 224])
x2 = torch.randn([2, 10, 3, 224, 224])
posenet_out = posenet(x1)
mapnet_out = mapnet(x2)

print("posenet_out size: {}".format(posenet_out.size()))
print("mapnet_out size: {}".format(mapnet_out.size()))
