import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.autograd.function import Function

from models_utils import *

class LeNet01(nn.Module):
    def __init__(self):
        super(LeNet01, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 64, 3)

        self.fc1 = nn.Linear(1024, 200)
        self.dropout1 = nn.Dropout(0.5, inplace=True)
        self.fc2 = nn.Linear(200, 200)
        self.dropout2 = nn.Dropout(0.5, inplace=True)
        self.fc3 = nn.Linear(200, 10)

    def forward(self, x):
        zeros = torch.zeros_like(x)
        ones = torch.ones_like(x)
        x = torch.where(x > 0, zeros, ones)
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
        x = self.dropout2(x)
        y = x
        x = self.fc3(x)

        return x, y

    def name(self):
        return "LeNet01"

class DefenseNet(nn.Module):
    def __init__(self):
        super(DefenseNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 64, 3)

        self.fc1 = nn.Linear(1024, 200)
        self.dropout1 = nn.Dropout(0.5, inplace=True)
        self.fc2 = nn.Linear(200, 200)
        self.dropout2 = nn.Dropout(0.5, inplace=True)

        # one layer
        self.ae1 = nn.Linear(200, 100)
        self.ae2 = nn.Linear(100, 200)

        self.fc3 = nn.Linear(200, 10)

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
        x = self.dropout2(x)

        x = self.ae1(x)
        x = F.relu(self.ae2(x))

        x = self.fc3(x)
        return x

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 64, 3)

        self.fc1 = nn.Linear(1024, 200)
        self.dropout1 = nn.Dropout(0.5, inplace=True)
        self.fc2 = nn.Linear(200, 200)
        self.dropout2 = nn.Dropout(0.5, inplace=True)
        self.fc3 = nn.Linear(200, 10)

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
        x = self.dropout2(x)
        x = self.fc3(x)

        return x


class LeNet_ANL(MNISTModel):
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
        return "LeNet_ANL"


class LeNet_Block(MNISTModel):
    def __init__(self):
        super(LeNet_Block, self).__init__()
        self.backblock = BackBlock.apply

    def forward(self, x):
        x = self.backblock(x)
        return super(LeNet_Block, self).forward(x)

    def name(self):
        return "LeNet_Block"
