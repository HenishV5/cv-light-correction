import torch
import torch.nn as nn


class Z_DCE_Network(nn.Module):
    def __init__(self):
        super(Z_DCE_Network, self).__init__()
        def convolution(in_channel, out_channel):
            return nn.Sequential(
                    nn.Conv2d(in_channel, out_channel, 3, 1, 1),
                    nn.ReLU(inplace=True)
            )
        self.conv1 = convolution(1, 32)
        self.conv2 = convolution(32, 32)
        self.conv3 = convolution(32, 32)
        self.conv4 = convolution(32, 32)
        self.conv5 = convolution(32, 32)
        self.conv6 = convolution(32, 32)
        self.conv7 = nn.Conv2d(32, 1, 3, 1, 1)
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        A = torch.tanh(self.conv7(x6 + x3))
        enhanced_lightning = x
        for _ in range(8):
            enhanced_lightning = enhanced_lightning + A * (enhanced_lightning ** 2 - enhanced_lightning)
        
        return enhanced_lightning


class SR_CNN(nn.Module):
    def __init__(self):
        super(SR_CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 9, 1, 4)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 32, 5, 1, 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(32, 3, 5, 1, 2)
    
    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.conv3(x)
        return x