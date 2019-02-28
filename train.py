import time
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from model.lanenet import LaneNet
from model.hnet import HNet
from model.loss import cross_entropy_loss, discriminative_loss, hnet_loss
from utils.dataset import LaneNetDataset, HNetDataset
from utils.config import CFG

def minmax_scale(input_arr):
    min_val = torch.min(input_arr)
    max_val = torch.max(input_arr)
    output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)

    return output_arr

def compute_lanenet_loss(net_output, binary_label, instance_label):
    # cross entropy loss of binary segmentation
    binary_seg_logits = net_output['binary_seg_logits']
    binary_loss = cross_entropy_loss(binary_seg_logits, binary_label)
    
    # discriminative loss os instance segmentation
    pixel_embedding = net_output['instance_seg_logits']
    feature_dim = pixel_embedding.size()[1]
    image_shape = (CFG.LANENET.IMAGE_H, CFG.LANENET.IMAGE_W)
    disc_loss, _, _, _ = discriminative_loss(pixel_embedding, instance_label, feature_dim, image_shape, CFG.LANENET.DELTA_V, CFG.LANENET.DELTA_D, CFG.LANENET.PARAM_VAR, CFG.LANENET.PARAM_DIST, CFG.LANENET.PARAM_REG)

    total_loss = 0.5 * binary_loss + 0.5 * disc_loss
    return total_loss, binary_loss, disc_loss

def train_lanenet(
        model, 
        dataset,
        lr=CFG.LANENET.LR, 
        batch_size=CFG.LANENET.BATCH_SIZE,
        epoch=CFG.EPOCH, 
        weights_path=CFG.DEFAULT_PATH):
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    print('data loaded')

    optimizer = optim.Adam(model.parameters(), lr=lr)
    print('optimizer inited')

    for i in range(epoch):
        print('Epoch {} start'.format(i+1))
        if i % 100 == 0:
            start = time.time()
        total_losses = []
        binary_losses = []
        disc_losses = []
        for batch_idx, batch_data in enumerate(data_loader):
            print('Batch {} start training'.format(batch_idx+1))
            img_data = batch_data['input_tensor']
            binary_label = batch_data['binary_label']
            instance_label = batch_data['instance_label']

            net_output = model(img_data)
            print('Batch {} forward finished'.format(batch_idx+1))
            binary_seg_pred = net_output['binary_seg_pred']
            total_loss, binary_loss, disc_loss = compute_lanenet_loss(net_output, binary_label, instance_label)
            total_losses.append(total_loss)
            binary_losses.append(binary_loss)
            disc_losses.append(disc_loss)

            print('Batch {bi} | total loss: {tl} | binary loss: {bl} | disc loss: {dl}'.format(bi=batch_idx+1, tl=total_loss, bl=binary_loss, dl=disc_loss))

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            print('Batch {} backward finished'.format(batch_idx+1))

        if i % 99 == 0:
            end = time.time()
            print('Epoch {ep}| running time: {rt} | total loss: {tl} | binary loss: {bl} | disc loss: {dl}'.format(ep=i+1, rt=end-start, tl=sum(total_losses)/len(total_losses), bl=sum(binary_losses)/len(binary_losses), dl=sum(instance_losses)/len(instance_losses)))
    
    model_name = weights_path + '_' + str(time.time()) + '_' + str(epoch) + '.pt'
    torch.save(model.state_dict(), model_name)

def train_hnet(
        model, 
        dataset,
        lr=CFG.HNET.LR, 
        batch_size=1,
        epoch=CFG.EPOCH, 
        weights_path=CFG.DEFAULT_PATH):
    data_loader = DataLoader(hnet_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    print('data loaded')

    optimizer = optim.Adam(model.parameters(), lr=lr)
    print('optimizer inited')

    for i in range(epoch):
        print('Epoch {} start'.format(i+1))
        if i % 100 == 0:
            start = time.time()
        losses = []
        for batch_idx, batch_data in enumerate(data_loader):
            print('Batch {} start training'.format(batch_idx+1))
            img_data = Variable(batch_data['input_tensor'])
            gt_points = Variable(batch_data['gt_points'])

            net_output = model(img_data).view(-1, 6)
            print('Batch {} forward finished'.format(batch_idx + 1))
            loss = hnet_loss(gt_points, net_output)
            losses.append(loss)
            print('Batch {bi} | total loss: {tl}'.format(bi=batch_idx+1, tl=loss))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('Batch {} backward finished'.format(batch_idx + 1))
        
        if i % 99 == 0:
            end = time.time()
            print('Epoch {ep}| running time: {rt} | total loss: {tl}'.format(ep=i+1, rt=end-start, tl=sum(losses)/len(losses)))

    model_name = weights_path + '_' + str(time.time()) + '_' + str(epoch) + '.pt'
    torch.save(model.state_dict(), model_name)

if __name__ == '__main__':
    # lanenet = LaneNet()
    # lanenet_dataset = LaneNetDataset('/Users/xujiale/Data/tusimple/train_set/training/train.txt', is_training=True)
    # train_lanenet(lanenet, lanenet_dataset, batch_size=4, epoch=1)

    hnet = HNet()
    hnet_dataset = HNetDataset([
        '/Users/xujiale/Data/tusimple/train_set/label_data_0313.json',
        '/Users/xujiale/Data/tusimple/train_set/label_data_0531.json',
        '/Users/xujiale/Data/tusimple/train_set/label_data_0601.json'], is_training=True)
    train_hnet(hnet, hnet_dataset, batch_size=1, epoch=1)