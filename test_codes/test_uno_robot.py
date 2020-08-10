import sys

sys.path.append('.')

import torch
from torch.utils.data import DataLoader
from mapnet.datasets import UnoRobot
from torch.utils.data.dataloader import default_collate
from torchvision import transforms


def safe_collate(batch):
    """
    Collate function for DataLoader that filters out None's
    :param batch: minibatch
    :return: minibatch filtered for None's
    """
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)


tforms = [transforms.Resize(256),
          transforms.ToTensor()]
data_transform = transforms.Compose(tforms)
target_transform = transforms.Lambda(lambda x: torch.from_numpy(x).float())

scene = 'corridor'
data_path = 'data/deepslam_data/UnoRobot/'
uno_robot = UnoRobot(scene=scene, data_path=data_path,
                     train=True, transform=data_transform,
                     target_transform=target_transform)

data_loader = DataLoader(uno_robot, batch_size=128, shuffle=True,
                         num_workers=0, collate_fn=safe_collate)

for batch_data, label in data_loader:
    print(batch_data.size(), label.size())
