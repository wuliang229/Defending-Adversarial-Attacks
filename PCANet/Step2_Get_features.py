import os
import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tempfile import TemporaryFile
from Step1_Train_LeNet import LeNet

use_cuda = torch.cuda.is_available()
outfile = TemporaryFile()
root = './data'
if not os.path.exists(root):
    os.mkdir(root)

trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
# if not exist, download mnist dataset
train_set = dset.MNIST(root=root, train=True, transform=trans, download=True)
test_set = dset.MNIST(root=root, train=False, transform=trans, download=True)

batch_size = 128

train_loader = torch.utils.data.DataLoader(
    dataset=train_set,
    batch_size=batch_size,
    shuffle=False)
test_loader = torch.utils.data.DataLoader(
    dataset=test_set,
    batch_size=batch_size,
    shuffle=False)

## training
if use_cuda:
    model = torch.load('LeNet.pth')
    model = model.cuda()
else:
    model = torch.load('LeNet.pth', map_location='cpu')
model.eval()

res = []
for batch_idx, (x, target) in enumerate(train_loader):
    if use_cuda:
        x, target = x.cuda(), target.cuda()
    out = model(x)[1].detach().cpu().numpy()
    for i in out:
        res.append(i)
res = np.array(res)
np.save('res.npy', res)
print(res.shape)

res = []
for batch_idx, (x, target) in enumerate(test_loader):
    if use_cuda:
        x, target = x.cuda(), target.cuda()
    out = model(x)[1].detach().cpu().numpy()
    for i in out:
        res.append(i)
res = np.array(res)
np.save('res_dev.npy', res)
print(res.shape)