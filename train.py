import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from easydict import EasyDict as edict

from model.loss import cross_entropy_loss, discriminative_loss, hnet_loss
from utils.dataset import LaneNetDataset, HNetDataset

CFG = edict()
CFG.EPOCH = 10000

CFG.LANENET = edict()
CFG.LANENET.LR = 5e-4
CFG.LANENET.BATCH_SIZE = 8
CFG.LANENET.IMAGE_W = 512
CFG.LANENET.IMAGE_H = 256
CFG.LANENET.DELTA_V = 0.5
CFG.LANENET.DELTA_D = 3.0
CFG.LANENET.PARAM_VAR = 1.0
CFG.LANENET.PARAM_DIST = 1.0
CFG.LANENET.PARAM_REG = 0.01

CFG.HNET = edict()
CFG.HNET.LR = 5e-5
CFG.HNET.BATCH_SIZE = 10
CFG.HNET.IMAGE_H = 128
CFG.HNET.IMAGE_W = 64

VGG_MEAN = [103.939, 116.779, 123.68]

def minmax_scale(input_arr):
    min_val = torch.min(input_arr)
    max_val = np.max(input_arr)
    output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)

    return output_arr

def compute_lanenet_loss(net_output, binary_label, instance_label):
    # cross entropy loss of binary segmentation
    binary_seg_logits = net_output['binary_seg_logits']
    binary_loss = cross_entropy_loss(binary_seg_logits, binary_label)
    
    # discriminative loss os instance segmentation
    pixel_embedding = net_output['instance_seg_logits']
    feature_dim = pixel_embedding.size()[0]
    image_shape = (CFG.LANENET.IMAGE_H, CFG.LANENET.IMAGE_W)
    disc_loss, _, _, _ = discriminative_loss(pixel_embedding, instance_label, feature_dim, image_shape, CFG.LANENET.DELTA_V, CFG.LANENET.DELTA_D, CFG.LANENET.PARAM_VAR, CFG.LANENET.PARAM_DIST, CFG.LANENET.PARAM_REG)

    total_loss = 0.5 * binary_loss + 0.5 * disc_loss
    return total_loss, binary_loss, disc_loss

def train_lanenet(
        model, 
        dataset_dir, 
        lr=CFG.LANENET.LR, 
        batch_size=CFG.LANENET.BATCH_SIZE, 
        epoch=CFG.EPOCH, 
        weights_path=None):
    train_dataset_file = os.path.join(dataset_dir, 'train.txt')
    assert os.path.exists(train_dataset_file)

    train_dataset = LaneNetDataset(train_dataset_file, is_training=True)
    data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    for i in range(epoch):
        if i % 100 == 0:
            start = time.time()
        total_losses = []
        binary_losses = []
        instance_losses = []
        start = time.time()
        for batch_idx, batch_data in enumerate(data_loader):
            img_data = Variable(batch_data['input_tensor'])
            binary_label = Variable(batch_data['binary_label'])
            instance_label = Variable(batch_data['instance_label'])

            net_output = model(img_data)
            binary_seg_pred = net_output['binary_seg_pred']
            total_loss, binary_loss, instance_loss = compute_lanenet_loss(net_output, binary_label, instance_label)
            total_losses.append(total_loss)
            binary_losses.append(binary_loss)
            instance_losses.append(instance_loss)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        end = time.time()
        if i % 99 == 0:
            end = time.time()
            print('Epoch {ep}| running time: {rt} | total loss: {tl} | binary loss: {bl} | instance loss: {il}'.format(ep=i+1, rt=end-start, tl=sum(total_losses)/len(data_loader), bl=sum(binary_losses)/len(data_loader), il=sum(instance_losses)/len(data_loader)))