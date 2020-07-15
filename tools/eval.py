"""
eval codes.
"""
import argparse
import os
import os.path as osp
import sys
import numpy as np
import torch.cuda
import pickle
from mmcv import Config
sys.path.append('.')
from mapnet.utils.pose_utils import quaternion_angular_error
from mapnet.apis.eval import build_pose_model, build_dataset_with_loader
from mapnet.apis.eval import inference, draw_result


def parse_args():
    # config
    parser = argparse.ArgumentParser(description='Evaluation script')
    parser.add_argument('--config_file', type=str, required=True,
                        help='configuration file')
    parser.add_argument('--weights', type=str, required=True,
                        help='trained weights to load')

    parser.add_argument('--device', type=str, default='0',
                        help='GPU device(s)')
    parser.add_argument('--val', action='store_true',
                        help='Plot graph for val')
    parser.add_argument('--show', action='store_true',
                        help='show the result or not')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output image directory')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    cfgs = Config.fromfile(args.config_file)

    # build model with cfg
    model_cfgs = cfgs.model
    CUDA = torch.cuda.is_available()
    model = build_pose_model(model_cfgs, args.weights, CUDA)
    model.eval()

    # build dataset with cfg
    data_cfgs = cfgs.dataset
    tform_cfgs = cfgs.transform
    data_cfgs.train = True if not args.val else False
    data_set, loader = build_dataset_with_loader(data_cfgs, tform_cfgs)
    data_name = data_cfgs.name if model_cfgs.type == 'MapNet' else data_cfgs.type

    # inference
    pred_poses, targ_poses = inference(model, data_set, loader, CUDA)
    print(pred_poses.shape)

    # loss functions
    t_criterion = lambda t_pred, t_gt: np.linalg.norm(t_pred - t_gt)
    q_criterion = quaternion_angular_error
    # calculate losses
    t_loss = np.asarray([t_criterion(p, t) for p, t in zip(pred_poses[:, :3],
                                                           targ_poses[:, :3])])
    q_loss = np.asarray([q_criterion(p, t) for p, t in zip(pred_poses[:, 3:],
                                                           targ_poses[:, 3:])])
    print('Error in translation: median {:3.2f} m,  mean {:3.2f} m\n' \
          'Error in rotation: median {:3.2f} degrees, mean {:3.2f} degree' \
          .format(np.median(t_loss), np.mean(t_loss), np.median(q_loss),
                  np.mean(q_loss)))

    dim3 = data_set.dim3 if hasattr(data_set, 'dim3') else False
    fig = draw_result(pred_poses, targ_poses, dim3, args.show)

    if args.output_dir is not None:
        experiment_name = '{:s}_{:s}_{:s}' \
            .format(data_name, data_cfgs.scene, model_cfgs.type)
        image_filename = osp.join(osp.expanduser(args.output_dir),
                                  '{:s}.png'.format(experiment_name))
        fig.savefig(image_filename)
        print('{:s} saved'.format(image_filename))
        result_filename = osp.join(osp.expanduser(args.output_dir), '{:s}.pkl'.
                          format(experiment_name))
        with open(result_filename, 'wb') as f:
            pickle.dump({'targ_poses': targ_poses, 'pred_poses': pred_poses}, f)
        print('{:s} written'.format(result_filename))


if __name__ == "__main__":
    main()
