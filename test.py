import time
import torch
from torch.utils.data import DataLoader

from model.lanenet import LaneNet
from model.hnet import HNet
from utils.dataset import LaneNetDataset, HNetDataset
from utils.config import CFG

def test_lanenet(model, dataset, batch_size=CFG.LANENET.BATCH_SIZE):
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    print('data loaded')

    for batch_idx, batch_data in enumerate(data_loader):
        start = time.time()
        print('Batch {} start testing'.format(batch_idx + 1))

        input_tensor = batch_data['input_tensor']
        net_output = model(input_tensor)
        print(net_output['instance_seg_logits'])

        end = time.time()

        print('Batch {bi} running time: {rt}'.format(bi=batch_idx + 1, rt=end-start))

def test_hnet(model, dataset, batch_size=1):
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    print('data loaded')

    for batch_idx, batch_data in enumerate(data_loader):
        start = time.time()
        print('Batch {} start testing'.format(batch_idx + 1))

        input_tensor = batch_data['input_tensor']
        net_output = model(input_tensor)
        print(net_output)

        end = time.time()

        print('Batch {bi} running time: {rt}'.format(bi=batch_idx + 1, rt=end-start))

if __name__ == '__main__':
    lanenet = LaneNet()
    lanenet_dataset = LaneNetDataset('/Users/xujiale/Data/tusimple/train_set/training/train.txt', is_training=False)
    test_lanenet(lanenet, lanenet_dataset, batch_size=4)

    hnet = HNet()
    hnet_dataset = HNetDataset([
        '/Users/xujiale/Data/tusimple/train_set/label_data_0313.json',
        '/Users/xujiale/Data/tusimple/train_set/label_data_0531.json',
        '/Users/xujiale/Data/tusimple/train_set/label_data_0601.json'], is_training=False)
    test_hnet(hnet, hnet_dataset, batch_size=1)