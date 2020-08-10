import sys
sys.path.append('.')
# from mapnet.models import ResNet34
from mapnet.builder import build_backbone


cfgs = {'type': 'ResNet34', 'pretrained': True}
resnet34 = build_backbone(cfgs)
print(resnet34)
