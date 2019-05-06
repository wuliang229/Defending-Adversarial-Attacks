import os
import torch
import copy
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm


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


if __name__ == '__main__':
    ## load mnist dataset
    use_cuda = torch.cuda.is_available()
    best_loss = 1e4
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

    ## training
    model = LeNet()
    if use_cuda:
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(20):
        # training
        model.train()
        train_loss = 0.
        loader = tqdm(train_loader, total=len(train_loader))
        for batch_idx, (x, target) in enumerate(loader):
            optimizer.zero_grad()
            if use_cuda:
                x, target = x.cuda(), target.cuda()
            out = model(x)[0]
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
                if use_cuda:
                    x, target = x.cuda(), target.cuda()
                out = model(x)[0]
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

        if test_loss < best_loss:
            best_model_wts = copy.deepcopy(model.state_dict())
            best_loss = test_loss

    # saving
    model.load_state_dict(best_model_wts)
    torch.save(model, 'LeNet.pth')
