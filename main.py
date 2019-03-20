import argparse
import os
from torch.backends import cudnn

from solver import Solver
from dataset import get_lanenet_loader, get_hnet_loader


def main(config):
    cudnn.benchmark = True

    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.model_dir):
        os.makedirs(config.model_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

    data_loader = None

    if config.mode == 'train':
        assert os.path.exists(config.dataset_dir)
        if config.net == 'lanenet':
            batch_size = 8 if config.batch_size is None else config.batch_size
            image_w = 512 if config.image_w is None else config.image_w
            image_h = 256 if config.image_h is None else config.image_h
            data_loader = get_lanenet_loader(config.dataset_dir, batch_size, image_w, image_h, is_training=True)
        else:
            batch_size = 10 if config.batch_size is None else config.batch_size
            image_w = 128 if config.image_w is None else config.image_w
            image_h = 64 if config.image_h is None else config.image_h
            data_loader = get_lanenet_loader(config.dataset_dir, batch_size, image_w, image_h, is_training=True)

    solver = Solver(config, data_loader)

    if config.mode == 'train':
        solver.train()
    else:
        solver.test()

def init_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='train or test')
    parser.add_argument('--dataset_dir', type=str, default='/root/datasets/tusimple/train_set/training', help='directory of tusimple dataset')

    parser.add_argument('--batch_size', type=int, default=None, help='training batch size')
    parser.add_argument('--image_w', type=int, default=None, help='image width')
    parser.add_argument('--image_h', type=int, default=None, help='image height')

    parser.add_argument('--net', type=str, default='lanenet', choices=['lanenet', 'hnet'], help='which net to be trained')
    parser.add_argument('--resume_iters', type=int, default=0, help='resume training from this step')
    parser.add_argument('--num_iters', type=int, default=20000, help='total training iterations')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--delta_v', type=float, default=0.5, help='delta v for computing discriminative loss of lanenet')
    parser.add_argument('--delta_d', type=float, default=3.0, help='delta d for computing discriminative loss of lanenet')
    parser.add_argument('--param_var', type=float, default=1.0, help='parameter var for computing discriminative loss of lanenet')
    parser.add_argument('--param_dist', type=float, default=1.0, help='parameter dist for computing discriminative loss of lanenet')
    parser.add_argument('--param_reg', type=float, default=0.01, help='parameter reg for computing discriminative loss of lanenet')

    parser.add_argument('--lanenet_path', type=str, help='path of lanenet model for testing')
    parser.add_argument('--hnet_path', type=str, help='path of hnet model for testing')
    parser.add_argument('--images_path', type=str, nargs='+', default=[], help='path of images for testing')
    parser.add_argument('--use_gpu', type=bool, default=False, help='use gpu or not')
    parser.add_argument('--show', type=bool, default=False, help='show the results of testing in plot or not')

    parser.add_argument('--gpus', type=int, nargs='+', default=[], help='gpu ids for training')

    parser.add_argument('--log_dir', type=str, default='logs', help='the directory to save logs in')
    parser.add_argument('--sample_dir', type=str, default='samples', help='the directory to save samples in')
    parser.add_argument('--model_dir', type=str, default='models', help='the directory to save model weights in')
    parser.add_argument('--result_dir', type=str, default='results', help='the directory to save testing results in')

    parser.add_argument('--log_step', type=int, default=10, help='save logs every so many steps')
    parser.add_argument('--sample_step', type=int, default=1000, help='save samples every so many steps')
    parser.add_argument('--model_save_step', type=int, default=5000, help='save model checkpoints every so many steps')

    return parser.parse_args()

if __name__ == '__main__':
    config = init_args()
    main(config)