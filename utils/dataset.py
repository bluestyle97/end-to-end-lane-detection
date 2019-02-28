import cv2
import json
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from utils.config import CFG

try:
    from cv2 import cv2
except ImportError:
    pass

class LaneNetDataset(Dataset):
    def __init__(self, info_file, is_training=True):
        self._is_training = is_training
        if self._is_training:
            self._img_list, self._binary_img_list, self._instance_img_list = self._init_dataset(info_file)
        else:
            self._img_list = self._init_dataset(info_file)
    
    def _init_dataset(self, info_file):
        assert os.path.exists(info_file), 'File {} does not exist.'.format(info_file)

        if self._is_training:
            img_list = []
            binary_img_list = []
            instance_img_list = []
            with open(info_file, 'r') as file:
                for line in file:
                    tmp = line.strip().split()
                    img_list.append(tmp[0])
                    binary_img_list.append(tmp[1])
                    instance_img_list.append(tmp[2])
            return img_list, binary_img_list, instance_img_list

        else:
            img_list = []
            with open(info_file, 'r') as file:
                for line in file:
                    tmp = line.strip().split()
                    img_list.append(tmp[0])
            return img_list

    def __len__(self):
        return len(self._img_list)

    def __getitem__(self, index):
        img_name = self._img_list[index]

        img = cv2.imread(img_name, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (512, 256), interpolation=cv2.INTER_LINEAR)
        img = np.asarray(img).astype(np.float32)
        img -= CFG.VGG_MEAN
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img)

        if not self._is_training:
            return {'input_tensor': img}

        binary_img_name = self._binary_img_list[index]
        instance_img_name = self._instance_img_list[index]

        binary_img = cv2.imread(binary_img_name, cv2.IMREAD_COLOR)
        binary_label = np.zeros([binary_img.shape[0], binary_img.shape[1]], dtype=np.uint8)
        idx = np.where((binary_img[:, :, :] != [0, 0, 0]).all(axis=2))
        binary_label[idx] = 1
        binary_label = cv2.resize(binary_label, (512, 256), interpolation=cv2.INTER_NEAREST)
        binary_label = torch.from_numpy(binary_label.reshape(1, 256, 512))

        instance_img = cv2.imread(instance_img_name, cv2.IMREAD_UNCHANGED)
        instance_label = cv2.resize(instance_img, (512, 256), interpolation=cv2.INTER_NEAREST)
        instance_label = torch.from_numpy(instance_label.reshape((1, 256, 512)))

        sample = {'input_tensor': img, 'binary_label': binary_label, 'instance_label': instance_label}
        return sample

class HNetDataset(Dataset):
    def __init__(self, json_files, is_training=True):
        self._is_training = is_training
        self._img_list, self._gt_pts_list = self._init_dataset(json_files)

    def _init_dataset(self, json_files):
        img_list = []
        gt_pts_list = []

        if self._is_training:
            for json_file in json_files:
                assert os.path.exists(json_file), 'File {} does not exist!'.format(json_file)

                src_dir = os.path.split(json_file)[0]
                with open(json_file, 'r') as file:
                    for line in file:
                        info_dict = json.loads(line)
                        img_path = os.path.join(src_dir, info_dict['raw_file'])
                        assert os.path.exists(img_path), 'File {} does not exist!'.format(img_path)

                        img_list.append(img_path)

                        h_samples = info_dict['h_samples']
                        lanes = info_dict['lanes']

                        lane_pts = []
                        for lane in lanes:
                            assert len(h_samples) == len(lane)
                            for x, y in zip(lane, h_samples):
                                if x == -2:
                                    continue
                                lane_pts.append([x, y, 1])
                            if not lane_pts:
                                continue
                            if len(lane_pts) <= 3:
                                continue
                        gt_pts_list.append(lane_pts)
            return img_list, gt_pts_list
        else:
            for json_file in json_files:
                assert os.path.exists(json_file), 'File {} does not exist!'.format(json_file)

                src_dir = os.path.split(json_file)[0]
                with open(json_file, 'r') as file:
                    for line in file:
                        info_dict = json.loads(line)
                        img_path = os.path.join(src_dir, info_dict['raw_file'])
                        assert os.path.exists(img_path), 'File {} does not exist!'.format(img_path)

                        img_list.append(img_path)
            return img_list

    def __len__(self):
        return len(self._img_list)

    def __getitem__(self, index):
        img_path = self._img_list[index]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (128, 64), interpolation=cv2.INTER_LINEAR)
        img = np.asarray(img).astype(np.float32)
        img -= CFG.VGG_MEAN
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img)

        if not self._is_training:
            return {'input_tensor': img}

        gt_pts = self._gt_pts_list[index]
        gt_pts = np.array(gt_pts).reshape((-1, 3))
        gt_pts = torch.from_numpy(gt_pts)

        sample = {'input_tensor': img, 'gt_points': gt_pts}
        
        return sample

if __name__ == '__main__':
    lanenet_dataset = LaneNetDataset('/Users/xujiale/Data/tusimple/train_set/training/train.txt', is_training=True)
    data_loader = DataLoader(lanenet_dataset, batch_size=4, shuffle=True, num_workers=0)
    for i, data in enumerate(data_loader):
        input_tensor = data['input_tensor']
        binary_label = data['binary_label']
        instance_label = data['instance_label']
        print(input_tensor, input_tensor.size())
        print(binary_label, binary_label.size())
        print(instance_label, instance_label.size())
        break

    hnet_dataset = HNetDataset([
        '/Users/xujiale/Data/tusimple/train_set/label_data_0313.json',
        '/Users/xujiale/Data/tusimple/train_set/label_data_0531.json',
        '/Users/xujiale/Data/tusimple/train_set/label_data_0601.json'], is_training=True)
    data_loader = DataLoader(hnet_dataset, batch_size=1, shuffle=True, num_workers=0)
    for data in data_loader:
        input_tensor = data['input_tensor']
        gt_points = data['gt_points']
        print(input_tensor, input_tensor.size())
        print(gt_points, gt_points.size())
        break