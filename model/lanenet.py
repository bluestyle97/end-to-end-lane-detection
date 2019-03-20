import torch
import torch.nn as nn
import torch.nn.functional as F


class VGG16Encoder(nn.Module):
    def __init__(self):
        super(VGG16Encoder, self).__init__()
        self.num_blocks = 5
        self.num_convs = [2, 2, 3, 3, 3]
        self.in_channels = [3, 64, 128, 256, 512]
        self.out_channels = [64, 128, 256, 512, 512]

        blocks = []
        for i in range(self.num_blocks):
            layers = []
            out_channels = self.out_channels[i]

            for j in range(self.num_convs[i]):
                in_channels = self.in_channels[i] if j == 0 else out_channels
                layers.append(nn.Conv2d(in_channels, out_channels, 3, padding=1))
                layers.append(nn.BatchNorm2d(out_channels))
                layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(2))

            block = nn.Sequential(*layers)
            blocks.append(block)

        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        ret = {}
        input = x
        for i, block in enumerate(self.net):
            output = block(input)
            ret['pool' + str(i + 1)] = output
            input = output
        return ret


class FCNDecoder(nn.Module):
    def __init__(self, decode_layers, in_channels):
        super(FCNDecoder, self).__init__()
        self.decode_layers = decode_layers
        self.in_channels = in_channels

        score_layers = []
        deconv_layers = []
        for i, c in enumerate(self.in_channels):
            score_layers.append(nn.Conv2d(c, 64, 1, bias=False))
            if i > 0:
                deconv_layers.append(nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1, bias=False))
        self.score_net = nn.Sequential(*score_layers)
        self.deconv_net = nn.Sequential(*deconv_layers)

        self.deconv_last = nn.ConvTranspose2d(64, 64, 16, stride=8, padding=4, bias=False)
        self.score_last = nn.Conv2d(64, 2, 1, bias=False)

    def forward(self, encoded_data):
        ret = {}
        for i, layer in enumerate(self.decode_layers):
            if i > 0:
                deconv = self.deconv_net[i - 1](score)
            input = encoded_data[layer]
            score = self.score_net[i](input)
            if i > 0:
                score = deconv + score
        deconv_final = self.deconv_last(score)
        score_final = self.score_last(deconv_final)
        ret['deconv'] = deconv_final
        ret['score'] = score_final
        return ret


class LaneNet(nn.Module):
    def __init__(self):
        super(LaneNet, self).__init__()
        self.encoder = VGG16Encoder()
        decode_layers = ['pool5', 'pool4', 'pool3']
        decode_in_channels = [512, 512, 256]
        self.decoder = FCNDecoder(decode_layers, decode_in_channels)
        self.pixel_layer = nn.Sequential(nn.Conv2d(64, 4, 1, bias=False), nn.ReLU())

    def forward(self, input):
        ret = {}
        encoder_ret = self.encoder(input)
        decoder_ret = self.decoder(encoder_ret)
        deconv = decoder_ret['deconv']
        score = decoder_ret['score']
        binary_seg_pred = torch.argmax(F.softmax(score, dim=1), dim=1, keepdim=True)
        pixel_embdding = self.pixel_layer(deconv)
        ret['instance_seg_logits'] = pixel_embdding
        ret['binary_seg_pred'] = binary_seg_pred
        ret['binary_seg_logits'] = score
        return ret


if __name__ == '__main__':
    model = LaneNet()
    input = torch.rand(4, 3, 256, 512, dtype=torch.float32)
    binary_label = torch.randint(1, (4, 1, 256, 512), dtype=torch.long)
    instance_label = torch.randint(4, (4, 1, 256, 512), dtype=torch.long)
    output = model(input)
    print(output['instance_seg_logits'].size())
    print(output['binary_seg_pred'].size())
    print(output['binary_seg_logits'].size())