"""
eval codes.
"""
import argparse
import os
import os.path as osp
import sys
import numpy as np
import pickle

sys.path.append('.')
from mapnet.apis import Inference
from mapnet.utils.yaml_utils import parse_yaml


def parse_args():
    # config
    parser = argparse.ArgumentParser(description='Evaluation script')
    parser.add_argument('--config', type=str, required=True,
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
    assert osp.exists(args.weights)
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    cfgs = parse_yaml(args.config)
    infer = Inference(cfgs, args.weights)
    pred_poses, targ_poses, fig = infer.eval_inference(args.val, args.show)

    t_loss, q_loss = infer.val_loss(pred_poses, targ_poses)
    print("Error in translation x : median {:3.2f} m,  mean {:3.2f} m"
          .format(np.median(t_loss[0]), np.mean(t_loss[0])))
    print("Error in translation y loss: median {:3.2f} m,  mean {:3.2f} m"
          .format(np.median(t_loss[1]), np.mean(t_loss[1])))
    print("Error in translation z loss: median {:3.2f} m,  mean {:3.2f} m"
          .format(np.median(t_loss[2]), np.mean(t_loss[2])))
    print("Error in translation loss: median {:3.2f} m,  mean {:3.2f} m"
          .format(np.median(t_loss[3]), np.mean(t_loss[3])))
    print("Error in rotation: median {:3.2f} degrees, mean {:3.2f} degree"
          .format(np.median(q_loss), np.mean(q_loss)))

    if args.output_dir is not None:
        image_file = osp.join(args.output_dir, 'result.png')
        fig.savefig(image_file)
        print('{:s} saved'.format(image_file))
        pose_file = osp.join(args.output_dir, 'result.pkl')
        with open(pose_file, 'wb') as f:
            pickle.dump({'targ_poses': targ_poses, 'pred_poses': pred_poses}, f)
        print('{:s} written'.format(pose_file))


if __name__ == "__main__":
    main()
