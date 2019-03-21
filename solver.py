import cv2
import datetime
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

from net.lanenet import LaneNet
from net.hnet import HNet
from net.loss import LanenetLoss, HNetLoss
from net.cluster import LaneNetCluster


VGG_MEAN = [103.939, 116.779, 123.68]

class Solver(object):
    def __init__(self, config, data_loader=None):
        # data loader and mode
        self.data_loader = data_loader
        self.mode = config.mode

        # training configurations
        self.net = config.net
        self.resume_iters = config.resume_iters
        self.num_iters = config.num_iters
        self.lr = config.lr
        self.delta_v = config.delta_v
        self.delta_d = config.delta_d
        self.param_var = config.param_var
        self.param_dist = config.param_dist
        self.param_reg = config.param_reg

        # testing configurations
        self.lanenet_path = config.lanenet_path
        self.hnet_path = config.hnet_path
        self.images_path = config.images_path
        self.use_gpu = config.use_gpu
        self.show = config.show

        # miscellaneous
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gpus = config.gpus

        # directories
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_dir = config.model_dir
        self.result_dir = config.result_dir

        # step size
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step

        self.build_model()

    def build_model(self):
        if self.mode == 'train':
            if self.net == 'lanenet':
                self.writer = SummaryWriter(os.path.join(self.log_dir, 'lanenet/'))
                self.lanenet = LaneNet()
                self.optimizer = optim.Adam(self.lanenet.parameters(), self.lr)

                self.lanenet.to(self.device)
                if torch.cuda.is_available() and len(self.gpus) > 1:
                    self.lanenet = nn.DataParallel(self.lanenet, device_ids=self.gpus)
            elif self.net == 'hnet':
                self.hnet = HNet()
                self.writer = SummaryWriter(os.path.join(self.log_dir, 'hnet/'))
                self.optimizer = optim.Adam(self.hnet.parameters(), self.lr)

                self.hnet.to(self.device)
                if torch.cuda.is_available() and len(self.gpus) > 1:
                    self.hnet = nn.DataParallel(self.hnet, device_ids=self.gpus)
        else:
            self.lanenet = LaneNet()
            self.hnet = HNet()

    def restore_model(self, net, model_path):
        assert net in ['lanenet', 'hnet']

        if net == 'lanenet':
            self.lanenet.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
        else:
            self.hnet.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))

    def compute_lanenet_loss(self, net_output, binary_label, instance_label):
        loss_fn = LanenetLoss()
        binary_seg_logits = net_output['binary_seg_logits']
        binary_loss = loss_fn.cross_entropy_loss(binary_seg_logits, binary_label)

        print('binary loss')

        pixel_embedding = net_output['instance_seg_logits']
        feature_dim = pixel_embedding.size()[1]
        disc_loss, _, _, _ = loss_fn.discriminative_loss(pixel_embedding, instance_label, feature_dim,
                                                         self.delta_v, self.delta_d, self.param_var,
                                                         self.param_dist, self.param_reg)
        print('discriminative loss')
        total_loss = 0.5 * binary_loss + 0.5 * disc_loss
        return total_loss, binary_loss, disc_loss

    def compute_hnet_loss(self, net_output, gt_points):
        loss_fn = HNetLoss()
        loss = loss_fn.hnet_loss(gt_points, net_output)
        return loss

    def minmax_scale(self, arr):
        min_val = np.min(arr)
        max_val = np.max(arr)

        output_arr = (arr - min_val) * 255.0 / (max_val - min_val)
        return output_arr

    def train(self):
        if self.net == 'lanenet':
            data_iter = iter(self.data_loader)

            start_iters = 0
            if self.resume_iters:
                start_iters = self.resume_iters
                model_path = os.path.join(self.model_dir, 'lanenet-{}.pt'.format(self.resume_iters))
                self.restore_model('lanenet', model_path)

            print('Start training...')
            start_time = time.time()
            for i in range(start_iters, self.num_iters):
                try:
                    batch_data = next(data_iter)
                except:
                    data_iter = iter(self.data_loader)
                    batch_data = next(data_iter)

                img_data = batch_data['input_tensor'].to(self.device)
                binary_label = batch_data['binary_label'].to(self.device)
                instance_label = batch_data['instance_label'].to(self.device)

                net_output = self.lanenet(img_data)
                total_loss, binary_loss, disc_loss = self.compute_lanenet_loss(net_output, binary_label, instance_label)

                print('loss computed')

                binary_seg_pred = net_output['binary_seg_pred']
                pixel_embedding = net_output['instance_seg_logits']

                acc = 0.
                batch_size = binary_seg_pred.size()[0]
                for i in range(batch_size):
                    PR = binary_seg_pred[i].squeeze().nonzero().size()[0]
                    GT = binary_label[i].squeeze().nonzero().size()[0]
                    TP = (binary_seg_pred[i].squeeze() * binary_label[i].squeeze()).nonzero().size()[0]
                    union = PR + GT - TP
                    acc += TP * 1.0 / union
                acc /= batch_size

                print('start backward')
                self.optimizer.zero_grad()
                total_loss.backward()
                print('end backward')
                self.optimizer.step()

                log = {}
                log['L/total_loss'] = total_loss.item()
                log['L/binary_loss'] = binary_loss.item()
                log['L/disc_loss'] = disc_loss.item()
                log['L/binary_seg_accuracy'] = acc

                if (i+1) % self.log_step == 0:
                    et = time.time() - start_time
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    info = 'Elapsed [{}], Iteration [{}/{}]'.format(et, i+1, self.num_iters)
                    for tag, value in log.items():
                        info += ', {}: {:.4f}'.format(tag, value)
                    print(info)

                    for tag, value in log.items():
                        self.writer.add_scalar(tag, value, i+1)

                if (i+1) % self.sample_step == 0:
                    step_sample_dir = os.path.join(self.sample_dir, '/step-{}/'.format(i+1))
                    if not os.path.exists(step_sample_dir):
                        os.makedirs(step_sample_dir)
                    cv2.imwrite(os.path.join(step_sample_dir, '{}_image.png'.format(i+1)),
                                img_data[0].cpu().numpy().transpose(1, 2, 0) + VGG_MEAN)
                    cv2.imwrite(os.path.join(step_sample_dir, '{}_binary_label.png'.format(i+1)),
                                binary_label[0].cpu().numpy().transpose(1, 2, 0) * 255)
                    cv2.imwrite(os.path.join(step_sample_dir, '{}_instance_label.png'.format(i+1)),
                                instance_label[0].cpu().numpy().transpose(1, 2, 0))
                    cv2.imwrite(os.path.join(step_sample_dir, '{}_binary_seg_pred.png'.format(i+1)),
                                binary_seg_pred[0].cpu().numpy().transpose(1, 2, 0) * 255)

                    embedding = pixel_embedding.cpu().numpy().transpose(1, 2, 0)
                    for i in range(4):
                        embedding[0][:, :, i] = self.minmax_scale(embedding[0][:, :, i])
                    embedding_image = np.array(embedding[0], np.uint8)
                    cv2.imwrite(os.path.join(step_sample_dir, '{}_pixel_embedding.png'.format(i+1)), embedding_image)
                    print('Save images into {}...'.format(step_sample_dir))

                if (i+1) % self.model_save_step == 0:
                    torch.save(self.lanenet.state_dict(), os.path.join(self.model_dir, 'lanenet-{}.pt'.format(i+1)))
                    print('Save net checkpoints into {}...'.format(self.model_dir))

        else:
            data_iter = iter(self.data_loader)

            start_iters = 0
            if self.resume_iters:
                start_iters = self.resume_iters
                model_path = os.path.join(self.model_dir, 'hnet-{}.pt'.format(self.resume_iters))
                self.restore_model('hnet', model_path)

            print('Start training...')
            start_time = time.time()
            for i in range(start_iters, self.num_iters):
                try:
                    batch_data = next(data_iter)
                except:
                    data_iter = iter(self.data_loader)
                    batch_data = next(data_iter)

                img_data = batch_data['input_tensor']
                gt_points = batch_data['gt_points']

                img_data.to(self.device, torch.float32)
                gt_points.to(self.device, torch.float32)

                net_output = self.hnet(img_data).view(-1, 6)
                loss = self.compute_hnet_loss(net_output, gt_points)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if (i+1) % self.log_step == 0:
                    et = time.time() - start_time
                    et = str(datetime.timedelta(seconds=et))[:-7]

                    info = 'Elapsed [{}], Iteration [{}/{}], H/loss: {:.4f}'.format(et, i+1, self.num_iters, loss)
                    print(info)

                    self.writer.add_scalar('H/loss', loss, i+1)

                if (i+1) % self.model_save_step == 0:
                    hnet_save_dir = os.path.join(self.model_dir, '/hnet/')
                    if not os.path.exists(hnet_save_dir):
                        os.makedirs(hnet_save_dir)
                    torch.save(self.hnet.state_dict(), os.path.join(hnet_save_dir, '{}-hnet.pt'.format(i+1)))
                    print('Save net checkpoints into {}...'.format(hnet_save_dir))

    def test(self):

        self.restore_model('lanenet', self.lanenet_path)
        self.restore_model('hnet', self.hnet_path)

        if torch.cuda.is_available() and self.use_gpu:
            self.lanenet.to(torch.device('cuda'))
            self.hnet.to(torch.device('cuda'))

        cluster = LaneNetCluster()

        for img_path in self.images_path:
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)
            image_lanenet = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
            image_lanenet = np.asarray(image_lanenet, dtype=np.float32)
            image_lanenet -= VGG_MEAN
            input_tensor_lanenet = torch.from_numpy(np.transpose(image_lanenet, (2, 0, 1)))
            if torch.cuda.is_available() and self.use_gpu:
                input_tensor_lanenet.to(torch.device('cuda'))

            image_hnet = cv2.resize(image, (128, 64), interpolation=cv2.INTER_LINEAR)
            image_hnet = np.asarray(image_hnet, dtype=np.float32)
            image_hnet -= VGG_MEAN
            input_tensor_hnet = torch.from_numpy(np.transpose(image_hnet, (2, 0, 1)))
            if torch.cuda.is_available() and self.use_gpu:
                input_tensor_hnet.to(torch.device('cuda'))

            with torch.no_grad():
                lanenet_output = self.lanenet(input_tensor_lanenet)
                binary_seg_pred = lanenet_output['binary_seg_pred']
                instance_seg_pred = lanenet_output['instance_seg_logits']

                binary_seg_image = binary_seg_pred.cpu().numpy().transpose((1, 2, 0))
                instance_seg_image = instance_seg_pred.cpu().numpy().transpose((1, 2, 0))

                mask_image = cluster.get_lane_mask(binary_seg_image, instance_seg_image)

                for i in range(4):
                    instance_seg_image[:, : i] = self.minmax_scale(instance_seg_image[:, : i])
                embedding_image = np.array(instance_seg_image, np.uint8)

                image_prefix = img_path.split('.')[0]
                cv2.imwrite(os.path.join(self.result_dir, image_prefix + '_src.png'), image)
                cv2.imwrite(os.path.join(self.result_dir, image_prefix + '_binary.png'), binary_seg_image * 255)
                cv2.imwrite(os.path.join(self.result_dir, image_prefix + '_instance.png'), embedding_image)
                cv2.imwrite(os.path.join(self.result_dir, image_prefix + '_mask.png'), mask_image)

                if self.show:
                    plt.figure('src_image')
                    plt.imshow(image[:, :, (2, 1, 0)])
                    plt.figure('binary_image')
                    plt.imshow(binary_seg_image[0] * 255, cmap='gray')
                    plt.figure('instance_image')
                    plt.imshow(embedding_image[:, :, (2, 1, 0)])
                    plt.figure('mask_image')
                    plt.imshow(mask_image[:, :, (2, 1, 0)])
                    plt.show()