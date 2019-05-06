import os
import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

use_cuda = torch.cuda.is_available()
if use_cuda:
    device = 'cuda'
else:
    device = 'cpu'

class CounterFrequency(nn.Module):
    def __init__(self):
        super(CounterFrequency, self).__init__()
        # models = []
        # for i in range(Nsteps):
        #     models.append(nn.Threshold((i + 1) / Nsteps - 1, 1 - i / (Nsteps - 1)))
        # self.through = nn.Sequential(*models)

        models0 = []
        models0.append(nn.Threshold(-0.75, 1.0))
        models0.append(nn.Threshold(-0.2, 0.5))
        models0.append(nn.Threshold(-0.0, 0.0))
        models1 = []
        models1.append(nn.Threshold(-0.75, 1.0))
        models1.append(nn.Threshold(-0.2, 0.5))
        models1.append(nn.Threshold(-0.0, 0.0))
        models2 = []
        models2.append(nn.Threshold(-0.6, 1.0))
        models2.append(nn.Threshold(-0.1, 0.5))
        models2.append(nn.Threshold(-0.0, 0.0))
        self.through0 = nn.Sequential(*models0)
        self.through1 = nn.Sequential(*models1)
        self.through2 = nn.Sequential(*models2)

    def forward(self, x):
        x = -x
        x[:, 0, :, :] = self.through0(x[:, 0, :, :])
        x[:, 1, :, :] = self.through1(x[:, 1, :, :])
        x[:, 2, :, :] = self.through2(x[:, 2, :, :])
        return x

if __name__ == '__main__':
    frequency = []
    num_pix = []
    keys = []
    frelist = []
    colors = ['y', 'b', 'k']
    for i in range(3):
        frequency.append(Counter())
        num_pix.append(0)
        keys.append(0)
        frelist.append(0)
    root = './data'
    if not os.path.exists(root):
        os.mkdir(root)
    # train_set = dset.MNIST(root=root, train=True, transform=transforms.ToTensor(), download=True)
    train_set = dset.CIFAR10(root='./data/CIFAR10', train=True, download=True, transform=transforms.ToTensor())
    batch_size = 128
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True)

    print('==>>> total trainning batch number: {}'.format(len(train_loader)))

    # to 0 and 1, N == 2
    Nsteps = 51
    model = CounterFrequency()
    if use_cuda:
        model = model.cuda()

    for batch_idx, (x, target) in enumerate(train_loader):
        if use_cuda:
            x, target = x.cuda(), target.cuda()
        out = model(x)
        # print(out.shape)
        out = out.transpose(0, 1)
        # print(out.shape)
        for i in range(len(out)):
            one_channel = out[i]
            # print(one_channel.shape)
            one_channel = one_channel.detach().cpu().contiguous().view(-1).numpy()
            tempFrequency = Counter(one_channel)
            frequency[i] += tempFrequency
        # break
    bar_width = 0.005
    for i in range(3):
        num_pix[i] = sum(frequency[i].values())
        keys[i] = list(frequency[i].keys())
        keys[i].sort()
        frelist[i] = []
        for j in keys[i]:
            print('value: ', j, round(frequency[i][j] / num_pix[i], 4))
            frelist[i].append(frequency[i][j] / num_pix[i])
        y_pos = np.linspace(0, 1, len(frelist[i]))
        plt.bar(y_pos + bar_width*i, frelist[i], color=colors[i], width=bar_width)
        plt.show()




