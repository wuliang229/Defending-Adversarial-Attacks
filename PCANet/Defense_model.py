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

batch_size = 128

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

        self.ae1 = nn.Linear(200, 100)
        self.ae2 = nn.Linear(100, 50)
        self.ae3 = nn.Linear(50, 100)
        self.ae4 = nn.Linear(100, 200)

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

        x = torch.tanh(self.ae1(x))
        x = self.ae2(x)
        x = torch.tanh(self.ae3(x))
        x = F.relu(self.ae4(x))

        x = self.fc3(x)
        return x

## training
model = torch.load('DefenseNet.pth')

if use_cuda:
    model = model.cuda()

criterion = nn.CrossEntropyLoss()

# testing
correct_cnt1, correct_cnt2 = 0, 0
total_cnt = 0
model.eval()
for batch_idx, (x, target) in enumerate(test_loader):
    if use_cuda:
        x, target = x.cuda(), target.cuda()
    out = model(x)
    loss2 = criterion(out, target)
    _, pred_label2 = torch.max(out.data, 1)
    total_cnt += x.data.size()[0]
    correct_cnt2 += torch.sum(torch.eq(pred_label2, target.data)).item()

    if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(test_loader):
        print('==>>> aec loss: {:.6f}, acc_aec: {:.3f}'.format(
               loss2,  correct_cnt2 * 1.0 / total_cnt))
        print('------------------------------------------------')


