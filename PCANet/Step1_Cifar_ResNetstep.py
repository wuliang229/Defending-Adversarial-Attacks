import os
import copy
import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.optim as optim
import torch
from torch.autograd.function import Function
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from tqdm import tqdm


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


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # params
    batch_size = 128
    num_epochs = 100
    best_accuracy = 0

    # dataset
    root = '../data'
    if not os.path.exists(root):
        os.mkdir(root)
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
    train_set = dset.CIFAR10(root=root, train=True, transform=transform, download=True)
    test_set = dset.CIFAR10(root=root, train=False, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

    print('==>>> total trainning batch number: {}'.format(len(train_loader)))
    print('==>>> total testing batch number: {}'.format(len(test_loader)))

    # model
    model = StepNet(BasicBlock, [2, 2, 2, 2])
    model = model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, cooldown=5,
                                                     min_lr=0.001, verbose=True)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        # training
        model.train()
        train_loss = 0.
        for batch_idx, (x, target) in enumerate(train_loader):
            optimizer.zero_grad()
            x, target = x.to(device), target.to(device)
            out = model(x)
            loss = criterion(out, target)
            loss = loss.mean()
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(train_loader):
                print('==>>> epoch: {}, batch index: {}, train loss: {:.6f}'.format(
                    epoch, batch_idx + 1, train_loss / (batch_idx + 1)))
        train_loss /= len(train_loader)

        # testing
        model.eval()
        correct_cnt, test_loss = 0, 0
        total_cnt = 0
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_loader):
                x, target = x.to(device), target.to(device)
                out = model(x)
                loss = criterion(out, target)
                loss = loss.mean()
                test_loss += loss.item()
                _, pred_label = torch.max(out.data, 1)
                total_cnt += x.data.size()[0]
                correct_cnt += torch.sum(torch.eq(pred_label, target.data)).item()

                if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(test_loader):
                    print('==>>> epoch: {}, batch index: {}, test loss: {:.6f}, acc: {:.3f}'.format(
                        epoch, batch_idx + 1, test_loss / (batch_idx + 1), correct_cnt * 1.0 / total_cnt))
                    print('------------------------------------------------')
            test_loss /= len(test_loader)
            accuracy = correct_cnt * 1.0 / total_cnt

        scheduler.step(accuracy)

        if accuracy > best_accuracy:
            best_model_wts = copy.deepcopy(model.state_dict())
            best_accuracy = accuracy
            torch.save(model, 'cifar_stepnet.pth')
