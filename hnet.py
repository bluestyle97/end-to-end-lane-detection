import torch
import torch.nn as nn

class HNet(nn.Module):
    def __init__(self):
        super(HNet, self).__init__()
        self._conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), 
            nn.BatchNorm2d(16), 
            nn.ReLU(inplace=True), 

            nn.Conv2d(16, 16, 3, padding=1), 
            nn.BatchNorm2d(16), 
            nn.ReLU(inplace=True), 

            nn.MaxPool2d(2), 

            nn.Conv2d(16, 32, 3, padding=1), 
            nn.BatchNorm2d(32), 
            nn.ReLU(inplace=True), 

            nn.Conv2d(32, 32, 3, padding=1), 
            nn.BatchNorm2d(32), 
            nn.ReLU(inplace=True), 

            nn.MaxPool2d(2), 

            nn.Conv2d(32, 64, 3, padding=1), 
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True), 

            nn.Conv2d(64, 64, 3, padding=1), 
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True), 

            nn.MaxPool2d(2))

        self._fc = nn.Sequential(
            nn.Linear(64 * 16 * 8, 1024), 
            nn.BatchNorm1d(1024), 
            nn.ReLU(inplace=True), 

            nn.Linear(1024, 6))
    
    def forward(self, x):
        x = self._conv(x)
        x = self._fc(x.view(-1, 64 * 16 * 8))
        return x
