import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.autograd.function import Function


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
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
        y = x
        x = self.fc3(x)

        return x, y

    def name(self):
        return "LeNet"


class LeNet_ANL(LeNet):
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
        y = x
        x = self.fc3(x)

        return x, y

    def name(self):
        return "LeNet_ANL"


class LeNet_Block(LeNet):
    def __init__(self):
        super(LeNet_Block, self).__init__()
        self.backblock = BackBlock.apply

    def forward(self, x):
        x = self.backblock(x)
        return super(LeNet_Block, self).forward(x)

    def name(self):
        return "LeNet_Block"


class ANL(nn.Module):
    def __init__(self):
        super(ANL, self).__init__()

    def forward(self, x, factor=0.02):
        if factor is not None:
            gaussian = x.new([Normal(factor / 2, factor / 4).sample()]).clamp(0, factor)
            std = torch.std(x)
            grad = x.grad
            noise = gaussian * std * grad / torch.norm(grad, float('inf'), keepdim=True)
        else:
            noise = 0
        return x + noise


class BackBlock(Function):
    def __init__(self):
        super(BackBlock, self).__init__()

    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = input.new(torch.zeros(size=input.shape))
        if ctx.needs_input_grad[1]:
            grad_weight = weight.new(torch.zeros(size=weight.shape))
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = bias.new(torch.zeros(size=bias.shape))

        return grad_input, grad_weight, grad_bias
