import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.autograd.function import Function

from models_utils import *

class CIFARModel01(nn.Module):
    def __init__(self):
        super(CIFARModel01, self).__init__()
        self.th1 = nn.Threshold(0.5, 0)
        self.th2 = nn.Threshold(-0.5, 1)
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv2 = nn.Conv2d(64, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv4 = nn.Conv2d(128, 128, 3)

        self.fc1 = nn.Linear(3200, 256)
        self.dropout1 = nn.Dropout(0.5, inplace=True)
        self.fc2 = nn.Linear(256, 256)
        # self.dropout2 = nn.Dropout(0.5, inplace = True)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.th1(x)
        x = -x
        x = self.th2(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2, 2)

        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        # x = self.dropout2(x)
        x = self.fc3(x)

        return x


class CIFARModel(nn.Module):
    def __init__(self):
        super(CIFARModel, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv2 = nn.Conv2d(64, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv4 = nn.Conv2d(128, 128, 3)

        self.fc1 = nn.Linear(3200, 256)
        self.dropout1 = nn.Dropout(0.5, inplace=True)
        self.fc2 = nn.Linear(256, 256)
        # self.dropout2 = nn.Dropout(0.5, inplace = True)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2, 2)

        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        # x = self.dropout2(x)
        x = self.fc3(x)

        return x

    def name(self):
        return 'LeNet_CIFAR'

class LeNet_ANL(CIFARModel):
    def __init__(self, noise_factor=None):
        super(LeNet_ANL, self).__init__()
        self.anl = ANL()
        self.noise_factor = noise_factor

    def forward(self, x, noise_factor=None):
        x = self.anl(F.relu(self.conv1(x)), self.noise_factor)
        x = self.anl(F.relu(self.conv2(x)), self.noise_factor)
        x = F.max_pool2d(x, 2, 2)
        x = self.anl(F.relu(self.conv3(x)), self.noise_factor)
        x = self.anl(F.relu(self.conv4(x)), self.noise_factor)
        x = F.max_pool2d(x, 2, 2)

        x = x.view(x.size(0), -1)  # flatten
        x = self.anl(F.relu(self.fc1(x)), self.noise_factor)
        x = self.dropout1(x)
        x = self.anl(F.relu(self.fc2(x)), self.noise_factor)
        x = self.dropout2(x)
        x = self.fc3(x)

        return x

    def name(self):
        return "LeNet_CIFAR_ANL"


class LeNet_Block(CIFARModel):
    def __init__(self):
        super(LeNet_Block, self).__init__()
        self.backblock = BackBlock.apply

    def forward(self, x):
        x = self.backblock(x)
        return super(LeNet_Block, self).forward(x)

    def name(self):
        return "LeNet_CIFAR_Block"
