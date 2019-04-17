import os
import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()

root = './data'
if not os.path.exists(root):
    os.mkdir(root)

trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
# if not exist, download mnist dataset
train_set = dset.MNIST(root=root, train=True, transform=trans, download=True)
test_set = dset.MNIST(root=root, train=False, transform=trans, download=True)

batch_size = 100

train_loader = torch.utils.data.DataLoader(
    dataset=train_set,
    batch_size=batch_size,
    shuffle=True)
test_loader = torch.utils.data.DataLoader(
    dataset=test_set,
    batch_size=batch_size,
    shuffle=False)

print('==>>> total trainning batch number: {}'.format(len(train_loader)))
print('==>>> total testing batch number: {}'.format(len(test_loader)))

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        y = x
        x = self.fc2(x)
        return x, y

    def name(self):
        return "LeNet"


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(500, 128),
            nn.Tanh(),
            nn.Linear(128, 64))
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 500),
            nn.ReLU())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

## training
model1 = torch.load('LeNet.pth')
model2 = torch.load('AEC.pth')


if use_cuda:
    model1 = model1.cuda()
    model2 = model2.cuda()

criterion = nn.CrossEntropyLoss()

# testing
correct_cnt1, correct_cnt2 = 0, 0
total_cnt = 0
model1.eval()
model2.eval()
for batch_idx, (x, target) in enumerate(test_loader):
    if use_cuda:
        x, target = x.cuda(), target.cuda()
    ori = model1(x)[0]
    out = model1(x)[1]
    out = model2(out)
    out = model1.fc2(out)

    loss1 = criterion(ori, target)
    loss2 = criterion(out, target)
    _, pred_label1 = torch.max(ori.data, 1)
    _, pred_label2 = torch.max(out.data, 1)
    total_cnt += x.data.size()[0]
    correct_cnt1 += torch.sum(torch.eq(pred_label1, target.data)).item()
    correct_cnt2 += torch.sum(torch.eq(pred_label2, target.data)).item()

    if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(test_loader):
        print('==>>> ori loss: {:.6f},aec loss: {:.6f}, acc_ori: {:.3f}, acc_aec: {:.3f}'.format(
              loss1, loss2, correct_cnt1 * 1.0 / total_cnt, correct_cnt2 * 1.0 / total_cnt))
        print('------------------------------------------------')


