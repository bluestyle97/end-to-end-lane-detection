import torch
from easydict import EasyDict as edict

config = edict()
config.epoch = 10000

config.lanenet = edict()
config.lanenet.lr = 5e-4
config.lanenet.batch_size = 8
config.lanenet.delta_v = 0.5
config.lanenet.delta_d = 3.0
config.lanenet.image_h = 512
config.lanenet.image_w = 256

config.hnet = edict()
config.hnet.lr = 5e-5
config.hnet.batch_size = 10
config.hnet.image_h = 128
config.hnet.image_w = 64
