import os
import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F

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

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(200, 100),
            nn.Tanh(),
            nn.Linear(100, 50))
        self.decoder = nn.Sequential(
            nn.Linear(50, 100),
            nn.Tanh(),
            nn.Linear(100, 200),
            nn.ReLU())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

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

model_le = torch.load('LeNet.pth')
model_ae = torch.load('AEC.pth')
dfnet = DefenseNet()

dfnet.conv1.weight = torch.nn.Parameter(model_le.state_dict()['conv1.weight'])
dfnet.conv1.bias = torch.nn.Parameter(model_le.state_dict()['conv1.bias'])
dfnet.conv2.weight = torch.nn.Parameter(model_le.state_dict()['conv2.weight'])
dfnet.conv2.bias = torch.nn.Parameter(model_le.state_dict()['conv2.bias'])
dfnet.conv3.weight = torch.nn.Parameter(model_le.state_dict()['conv3.weight'])
dfnet.conv3.bias = torch.nn.Parameter(model_le.state_dict()['conv3.bias'])
dfnet.conv4.weight = torch.nn.Parameter(model_le.state_dict()['conv4.weight'])
dfnet.conv4.bias = torch.nn.Parameter(model_le.state_dict()['conv4.bias'])
dfnet.fc1.weight = torch.nn.Parameter(model_le.state_dict()['fc1.weight'])
dfnet.fc1.bias = torch.nn.Parameter(model_le.state_dict()['fc1.bias'])
dfnet.fc2.weight = torch.nn.Parameter(model_le.state_dict()['fc2.weight'])
dfnet.fc2.bias = torch.nn.Parameter(model_le.state_dict()['fc2.bias'])
dfnet.fc3.weight = torch.nn.Parameter(model_le.state_dict()['fc3.weight'])
dfnet.fc3.bias = torch.nn.Parameter(model_le.state_dict()['fc3.bias'])

dfnet.ae1.weight = torch.nn.Parameter(model_ae.state_dict()['encoder.0.weight'])
dfnet.ae1.bias = torch.nn.Parameter(model_ae.state_dict()['encoder.0.bias'])
dfnet.ae2.weight = torch.nn.Parameter(model_ae.state_dict()['encoder.2.weight'])
dfnet.ae2.bias = torch.nn.Parameter(model_ae.state_dict()['encoder.2.bias'])

dfnet.ae3.weight = torch.nn.Parameter(model_ae.state_dict()['decoder.0.weight'])
dfnet.ae3.bias = torch.nn.Parameter(model_ae.state_dict()['decoder.0.bias'])
dfnet.ae4.weight = torch.nn.Parameter(model_ae.state_dict()['decoder.2.weight'])
dfnet.ae4.bias = torch.nn.Parameter(model_ae.state_dict()['decoder.2.bias'])

torch.save(dfnet, 'DefenseNet.pth')












