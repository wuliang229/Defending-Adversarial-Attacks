import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.autograd.function import Function

from models_utils import *


class StepNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(StepNet, self).__init__()
        self.in_planes = 64

        models0 = []
        thresholds = [0.161, 0.259, 0.341, 0.416, 0.482, 0.553, 0.631, 0.718, 0.839][::-1]
        for index, threshold in enumerate(thresholds):
            models0.append(nn.Threshold(-threshold, (len(thresholds) - 1 - index) / (len(thresholds) - 1)))

        models1 = []
        thresholds = [0.161, 0.255, 0.333, 0.404, 0.475, 0.541, 0.616, 0.702, 0.824][::-1]
        for index, threshold in enumerate(thresholds):
            models1.append(nn.Threshold(-threshold, (len(thresholds) - 1 - index) / (len(thresholds) - 1)))

        models2 = []
        thresholds = [0.122, 0.2, 0.271, 0.337, 0.408, 0.49, 0.584, 0.698, 0.843][::-1]
        for index, threshold in enumerate(thresholds):
            models2.append(nn.Threshold(-threshold, (len(thresholds) - 1 - index) / (len(thresholds) - 1)))

        self.through0 = nn.Sequential(*models0)
        self.through1 = nn.Sequential(*models1)
        self.through2 = nn.Sequential(*models2)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = -x
        x[:, 0, :, :] = self.through0(x[:, 0, :, :])
        x[:, 1, :, :] = self.through1(x[:, 1, :, :])
        x[:, 2, :, :] = self.through2(x[:, 2, :, :])
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def name(self):
        return "StepNet"



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


'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def name(self):
        return "cifar_resnet18"


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])
