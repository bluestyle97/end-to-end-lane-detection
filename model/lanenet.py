import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class VGG16Encoder(nn.Module):
    def __init__(self):
        super(VGG16Encoder, self).__init__()
        self._num_blocks = 5
        self._num_convs = [2, 2, 3, 3, 3]
        self._in_channels = [3, 64, 128, 256, 512]
        self._out_channels = [64, 128, 256, 512, 512]
        self._net = nn.Sequential()
        for i in range(self._num_blocks):
            self._net.add_module('block{}'.format(i+1), self._vgg_block(i+1))
    
    def _vgg_block(self, blk_id):
        block = nn.Sequential()
        out_channels = self._out_channels[blk_id-1]

        for i in range(self._num_convs[blk_id-1]):
            in_channels = self._in_channels[blk_id-1] if i == 0 else out_channels
            block.add_module('block{}_conv{}'.format(blk_id, i+1), nn.Conv2d(in_channels, out_channels, 3, padding=1))
            block.add_module('block{}_bn{}'.format(blk_id, i+1), nn.BatchNorm2d(out_channels))
            block.add_module('block{}_relu{}'.format(blk_id, i+1), nn.ReLU(inplace=True))
        block.add_module('block{}_pool'.format(blk_id), nn.MaxPool2d(2))

        return block

    def forward(self, x):
        ret = OrderedDict()
        input = x
        for i, block in enumerate(self._net):
            output = block(input)
            ret['pool'+str(i+1)] = output
            input = output
        return ret

class FCNDecoder(nn.Module):
    def __init__(self, decode_layers, in_channels):
        super(FCNDecoder, self).__init__()
        self._decode_layers = decode_layers
        self._in_channels = in_channels
        self._score_net = nn.Sequential()
        self._deconv_net = nn.Sequential()
        for i, c in enumerate(self._in_channels):
            self._score_net.add_module('conv{}'.format(i+1), nn.Conv2d(c, 64, 1, bias=False))
            if i > 0:
                self._deconv_net.add_module('deconv{}'.format(i), nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1, bias=False))
        self._deconv_last = nn.ConvTranspose2d(64, 64, 16, stride=8, padding=4, bias=False)
        self._score_last = nn.Conv2d(64, 2, 1, bias=False)
    
    def forward(self, encoded_data):
        ret = {}
        for i, layer in enumerate(self._decode_layers):
            if i > 0:
                deconv = self._deconv_net[i-1](score)
                # print('deconv{} size:'.format(i), deconv.size())
            input = encoded_data[layer]
            # print('input{} size:'.format(i), input.size())
            score = self._score_net[i](input)
            if i > 0:
                score = deconv + score
            # print('score{} size:'.format(i), score.size())
        deconv_final = self._deconv_last(score)
        # print('deconv final:', deconv_final.size())
        score_final = self._score_last(deconv_final)
        # print('score final:', score_final.size())
        ret['deconv'] = deconv_final
        ret['score'] = score_final
        return ret

class LaneNet(nn.Module):
    def __init__(self):
        super(LaneNet, self).__init__()
        self._encoder = VGG16Encoder()
        decode_layers = ['pool5', 'pool4', 'pool3']
        decode_in_channels = [512, 512, 256]
        self._decoder = FCNDecoder(decode_layers, decode_in_channels)
        self._pixel_layer = nn.Sequential(nn.Conv2d(64, 4, 1, bias=False), nn.ReLU())
    
    def forward(self, input):
        ret = {}
        encoder_ret = self._encoder(input)
        decoder_ret = self._decoder(encoder_ret)
        deconv = decoder_ret['deconv']
        score = decoder_ret['score']
        binary_seg_pred = torch.argmax(F.softmax(score, dim=1), dim=1, keepdim=True)
        pixel_embdding = self._pixel_layer(deconv)
        ret['instance_seg_logits'] = pixel_embdding
        ret['binary_seg_pred'] = binary_seg_pred
        ret['binary_seg_logits'] = score
        return ret

if __name__ == '__main__':
    model = LaneNet()
    input = torch.rand(4, 3, 256, 512, dtype=torch.float32)
    binary_label =torch.randint(1, (4, 1, 256, 512), dtype=torch.long)
    instance_label = torch.randint(4, (4, 1, 256, 512), dtype=torch.long)
    output = model(input)
    print(output['instance_seg_logits'].size())
    print(output['binary_seg_pred'].size())
    print(output['binary_seg_logits'].size())