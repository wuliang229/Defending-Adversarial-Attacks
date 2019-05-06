import os
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class TeacherData(Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index][0], torch.tensor(self.y[index])


class CIFARModel01(nn.Module):
    def __init__(self):
        super(CIFARModel01, self).__init__()
        self.th1 = nn.Threshold(0.5, 0)
        self.th2 = nn.Threshold(-0.5, 1)
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
        x = self.th1(x)
        x = -x
        x = self.th2(x)
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


def continuous_cross_entropy(logits, target):
    # (batch_size, 10)
    return - torch.sum(F.log_softmax(logits, dim=1) * target) / len(logits)


def train(loader, test_loader, file_name, num_epochs, is_student, train_temp=1):
    model = CIFARModel01()
    model.to(device)
    CE = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3)

    best_test_accuracy = 0

    for epoch in range(num_epochs):
        model.train()
        start = time.time()
        avg_loss = 0.0
        for batch_num, (x, target) in enumerate(loader):
            x = x.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            logits = model(x)
            logits /= train_temp

            # batch_loss = None
            if is_student:
                batch_loss = continuous_cross_entropy(logits, target)
            else:
                batch_loss = CE(logits, target)

            batch_loss.backward()
            optimizer.step()

            avg_loss += batch_loss.item()

            if batch_num % 100 == 99:
                print("Epoch {}, batch {}: average loss = {}".format(epoch + 1, batch_num + 1, avg_loss / 100))
                avg_loss = 0.0

            torch.cuda.empty_cache()
            del x
            del target
            del logits
            del batch_loss

        end = time.time()
        print("Epoch {} finished: total time = {} seconds".format(epoch + 1, end - start))
        accuracy = devmodel(model, test_loader)

        if accuracy > best_test_accuracy:
            print("Get a better testing accuracy, saving....")
            torch.save(model, file_name)
            best_test_accuracy = accuracy

        scheduler.step(accuracy)

    return model


def devmodel(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    for batch_num, (x, target) in enumerate(test_loader):
        x = x.to(device)
        target = target.to(device)
        logits = model(x)
        predictions = torch.argmax(logits, dim=1)
        correct += torch.sum(predictions == target).item()
        total += len(x)

        torch.cuda.empty_cache()
        del x
        del target
        del logits
        del predictions

    accuracy = correct / total
    print("Test accuracy = {}".format(accuracy))
    return accuracy

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CIFAR_trainset = datasets.CIFAR10(root='./data/CIFAR10', train=True, download=True, transform=transforms.ToTensor())
    CIFAR_testset = datasets.CIFAR10(root='./data/CIFAR10', train=False, download=True, transform=transforms.ToTensor())

    batch_size = 128
    num_epochs = 50

    train_loader = DataLoader(dataset=CIFAR_trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=CIFAR_testset, batch_size=batch_size, shuffle=False)

    # Regular training
    train(train_loader, test_loader, "LeNet_CIFAR1001.pth", num_epochs, False)
