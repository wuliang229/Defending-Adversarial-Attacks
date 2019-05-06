import os
import copy
import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.optim as optim

from tqdm import tqdm
from models_cifar import *

device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')

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
model = ResNet18()
model = model.to(device)

optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, cooldown=5,
                                                 min_lr=0.001, verbose=True)
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    # training
    model.train()
    train_loss = 0.
    loader = tqdm(train_loader, total=len(train_loader))
    for batch_idx, (x, target) in enumerate(loader):
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
    loader = tqdm(test_loader, total=len(test_loader))
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(loader):
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
        torch.save(model, os.path.join('..', 'model', model.name() + '.pth'))
