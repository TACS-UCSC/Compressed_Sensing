import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class CNN2D(nn.Module):
    def __init__(self, channelsin = 3, channelsout = 3):
        super(CNN2D, self).__init__()

        self.lrelu1 = nn.LeakyReLU()
        self.lrelu2 = nn.LeakyReLU()
        self.lrelu3 = nn.LeakyReLU()
        self.conv1 = nn.Conv2d(channelsin, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, channelsout, (3, 3), (1, 1), (1, 1))
        self._initialize_weights()

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.lrelu1(self.conv1(x))
        x = self.lrelu2(self.conv2(x))
        x = self.lrelu3(self.conv3(x))
        x = self.conv4(x)
        x = x.permute(0, 2, 3, 1)
        return x

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('leaky_relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('leaky_relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('leaky_relu'))
        init.orthogonal_(self.conv4.weight)
