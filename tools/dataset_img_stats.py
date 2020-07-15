"""
Computes the mean and std of pixels in a dataset (used in config file)
"""
import os.path as osp
import numpy as np
import argparse
import sys
sys.path.append('.')

from mapnet.builder import build_dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from mapnet.apis.train import safe_collate

DATA_DIR = 'data/deepslam_data/'


def parse_args():
    parser = argparse.ArgumentParser(description='Dataset images statistics')
    parser.add_argument('--dataset', type=str, choices=('SevenScenes', 'UnoRobot'),
                        help='Dataset', required=True)
    parser.add_argument('--scene', type=str,
                        help='Scene name', required=True)
    args = parser.parse_args()
    return args


def img_stats(dataset, scene, resize_min, crop_size, batch_size=8):
    crop_size = tuple(crop_size)
    data_transform = transforms.Compose([transforms.Resize(resize_min),
                                        transforms.CenterCrop(crop_size),
                                        transforms.ToTensor()])
    # dataset loader
    data_dir = osp.join(DATA_DIR, dataset)
    kwargs = dict(scene=scene, data_path=data_dir, train=True,
                  transform=data_transform, mean_t=np.zeros(3),
                  std_t=np.ones(3))
    kwargs['type'] = dataset
    dset = build_dataset(kwargs)

    # accumulate
    num_workers = batch_size
    loader = DataLoader(dset, batch_size=batch_size, num_workers=num_workers,
                        collate_fn=safe_collate)
    acc = np.zeros((3, crop_size[0], crop_size[1]))
    sq_acc = np.zeros((3, crop_size[0], crop_size[1]))
    for batch_idx, (imgs, _) in enumerate(loader):
        imgs = imgs.numpy()
        acc += np.sum(imgs, axis=0)
        sq_acc += np.sum(imgs**2, axis=0)
        if batch_idx % 50 == 0:
            print('Accumulated {:d} / {:d}'.format(batch_idx*batch_size, len(dset)))
    N = len(dset) * acc.shape[1] * acc.shape[2]
    mean_p = np.asarray([np.sum(acc[c]) for c in range(3)])
    mean_p /= N
    # std = E[x^2] - E[x]^2
    var_p = np.asarray([np.sum(sq_acc[c]) for c in range(3)])
    var_p /= N
    var_p -= (mean_p ** 2)

    return mean_p, var_p


if __name__ == "__main__":
    args = parse_args()
    resize_min = 256
    crop_size = tuple([224, 224])
    mean_p, var_p = img_stats(dataset=args.dataset, scene=args.scene,
                              resize_min=resize_min, crop_size=crop_size)
    print('Mean pixel = ', mean_p)
    print('Var. pixel = ', var_p)
    print('Std. pixel = ', np.sqrt(var_p))
