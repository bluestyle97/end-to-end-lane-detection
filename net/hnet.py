import torch
import torch.nn as nn

class HNet(nn.Module):
    def __init__(self):
        super(HNet, self).__init__()
        self.num_blocks = 3
        self.in_channels = [3, 16, 32]
        self.out_channels = [16, 32, 64]

        layers = []
        for i in range(self.num_blocks):
            in_channels = self.in_channels[i]
            out_channels = self.out_channels[i]
            layers.append(nn.Conv2d(in_channels, out_channels, 3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))

            layers.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))

            layers.append(nn.MaxPool2d(2))

        self.conv = nn.Sequential(*layers)
        self.fc = nn.Sequential(nn.Linear(64 * 16 * 8, 1024), nn.ReLU(inplace=True), nn.Linear(1024, 6))
    
    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x.view(-1, 64 * 16 * 8))
        return x

if __name__ == '__main__':
    hnet = HNet()
    input = torch.rand(4, 3, 64, 128, dtype=torch.float32)
    output = hnet(input)
    print(output.size())