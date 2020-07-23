"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import os.path as osp
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch.cuda
from torch.utils.data import DataLoader

from mapnet.datasets import build_transforms
from mapnet.builder import build_backbone, build_dataset, build_model
from mapnet.apis.train import load_state_dict, step_feedfwd
from mapnet.utils.pose_utils import qexp
from mapnet.utils.pose_utils import quaternion_angular_error as qae


class Inference(object):
    def __init__(self, cfgs, weights):
        self.cfgs = cfgs
        self.model_cfgs = cfgs.model
        self.CUDA = torch.cuda.is_available()
        self.model = self._build_val_model(cfgs.model, weights)
        self.model.eval()
        self.data_cfgs = cfgs.dataset
        self.tform_cfgs = cfgs.transform
        self.tform_cfgs.color_jitter = -1  # ignore the color jitter

    def _build_val_model(self, model_cfgs, weights):
        backbone_cfg = model_cfgs.backbone
        backbone_cfg.pretrained = False
        backbone = build_backbone(backbone_cfg)
        model_cfgs.backbone = backbone
        model_cfgs.pretrained = False
        model = build_model(model_cfgs)

        weights = osp.expanduser(weights)
        if osp.isfile(weights):
            loc_func = lambda storage, loc: storage
            checkpoint = torch.load(weights, map_location=loc_func)
            load_state_dict(model, checkpoint['model_state_dict'])
            print('Loaded weights from {:s}'.format(weights))
        else:
            print('Could not load weights from {:s}'.format(weights))
            sys.exit(-1)

        if self.CUDA:
            model.cuda()
        return model

    def _build_val_dataset(self, data_cfgs, tform_cfgs, val=True,
                           with_loader=True, num_workers=5):
        
        data_trans, target_trans = build_transforms(**tform_cfgs)

        data_cfgs.train = True if not val else False
        data_cfgs.transform = data_trans
        data_cfgs.target_transform = target_trans
        data_set = build_dataset(data_cfgs)

        loader = None
        if with_loader:
            loader = DataLoader(data_set, batch_size=1, shuffle=False,
                                num_workers=num_workers, pin_memory=True)
        return data_set, loader

    def draw_result(self, pred_poses, targ_poses, dim3=True, show=False):
        fig = plt.figure()
        if not dim3:
            ax = fig.add_subplot(111)
        else:
            ax = fig.add_subplot(111, projection='3d')
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1)

        # plot on the figure object
        L = targ_poses.shape[0]
        ss = max(1, int(L / 1000))  # 100 for stairs
        # scatter the points and draw connecting line
        x = np.vstack((pred_poses[::ss, 0].T, targ_poses[::ss, 0].T))
        y = np.vstack((pred_poses[::ss, 1].T, targ_poses[::ss, 1].T))
        if not dim3:  # 2D drawing
            ax.plot(x, y, c='b')
            ax.scatter(x[0, :], y[0, :], c='r')
            ax.scatter(x[1, :], y[1, :], c='g')
        else:
            z = np.vstack((pred_poses[::ss, 2].T, targ_poses[::ss, 2].T))
            for xx, yy, zz in zip(x.T, y.T, z.T):
                ax.plot(xx, yy, zs=zz, c='b')
            ax.scatter(x[0, :], y[0, :], zs=z[0, :], c='r', depthshade=0)
            ax.scatter(x[1, :], y[1, :], zs=z[1, :], c='g', depthshade=0)
            ax.view_init(azim=119, elev=13)
        if show:
            plt.show(block=True)
        return fig

    def eval_inference(self, val=True, show=False):
        data_set, loader = self._build_val_dataset(self.data_cfgs,
                                                   self.tform_cfgs,
                                                   val=val,
                                                   with_loader=True)
        L = len(data_set)
        pred_poses = np.zeros((L, 7))  # store all predicted poses
        targ_poses = np.zeros((L, 7))  # store all target poses
        pose_m = self.data_cfgs.mean_t
        pose_s = self.data_cfgs.std_t
        for batch_idx, (data, target) in enumerate(loader):
            if batch_idx % 200 == 0:
                print('Image {:d} / {:d}'.format(batch_idx, len(loader)))

            # indices into the global arrays storing poses
            idx = [batch_idx]
            idx = idx[len(idx) // 2]

            # output : 1 x 6 or 1 x STEPS x 6
            _, output = step_feedfwd(data, self.model, self.CUDA, train=False)
            s = output.size()
            output = output.cpu().data.numpy().reshape((-1, s[-1]))
            target = target.numpy().reshape((-1, s[-1]))

            # normalize the predicted quaternions
            q = [qexp(p[3:]) for p in output]
            output = np.hstack((output[:, :3], np.asarray(q)))
            q = [qexp(p[3:]) for p in target]
            target = np.hstack((target[:, :3], np.asarray(q)))

            # un-normalize the predicted and target translations
            output[:, :3] = (output[:, :3] * pose_s) + pose_m
            target[:, :3] = (target[:, :3] * pose_s) + pose_m

            # take the middle prediction
            pred_poses[idx, :] = output[len(output)//2]
            targ_poses[idx, :] = target[len(target)//2]

        print('Image {:d} / {:d}'.format(len(loader), len(loader)))

        dim3 = data_set.dim3 if hasattr(data_set, 'dim3') else False
        fig = self.draw_result(pred_poses, targ_poses, dim3, show=show)

        return pred_poses, targ_poses, fig

    def val_loss(self, pred_poses, targ_poses):
        def t_criterion(t_pred, t_gt):
            return np.linalg.norm(t_pred - t_gt)

        t_loss = np.asarray([t_criterion(p, t) for p, t in
                            zip(pred_poses[:, :3], targ_poses[:, :3])])
        q_loss = np.asarray([qae(p, t) for p, t in
                            zip(pred_poses[:, 3:], targ_poses[:, 3:])])

        return t_loss, q_loss

    def test_image(self, img):
        pred_pose = np.zeros(7)
        pose_m = np.array(self.data_cfgs.mean_t)
        pose_s = np.array(self.data_cfgs.std_t)
        data_trans, _ = build_transforms(**self.tform_cfgs)
        img = data_trans(img)
        if self.data_cfgs.type == 'MF':  # only for mapnet
            img = img.unsqueeze(0)  # step dim
        img = img.unsqueeze(0)  # batch dim
        _, output = step_feedfwd(img, self.model, self.CUDA, train=False)
        output = output.cpu().squeeze().numpy()
        pred_pose[:3] = (output[:3] * pose_s) + pose_m
        pred_pose[3:] = qexp(output[3:])
        return pred_pose
