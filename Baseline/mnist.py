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


class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 64, 3)

        self.fc1 = nn.Linear(1024, 200)
        self.dropout1 = nn.Dropout(0.5, inplace = True)
        self.fc2 = nn.Linear(200, 200)
        self.dropout2 = nn.Dropout(0.5, inplace = True)
        self.fc3 = nn.Linear(200, 10)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2, 2)

        x = x.view(x.size(0), -1) # flatten
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)

        return x


def continuous_cross_entropy(logits, target):
    # (batch_size, 10)
    return - torch.sum(F.log_softmax(logits, dim = 1) * target) / len(logits)


def train(loader, file_name, num_epochs, is_student, train_temp = 1):

    model = MNISTModel()
    model.to(device)
    CE = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
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
        accuracy = test(model)
        scheduler.step(accuracy)


    torch.save(model.state_dict(), 'models/' + file_name)
    return model


def test(model):
    model.eval()
    correct = 0
    total = 0
    for batch_num, (x, target) in enumerate(test_loader):
        x = x.to(device)
        target = target.to(device)
        logits = model(x)
        predictions = torch.argmax(logits, dim = 1)
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


def train_distillation(loader, file_name, num_epochs, train_temp = 1):

    print("Start training the teacher")
    teacher = train(loader, "teacher_" + file_name, num_epochs, False, train_temp)

    # evaluate the labels at temperature t
    new_labels = []
    teacher.eval()
    deterministic_loader = DataLoader(dataset=mnist_trainset, batch_size=batch_size, shuffle=False)
    
    print("Start generating the probabilities as labels")
    for batch_num, (x, target) in enumerate(deterministic_loader):
        x = x.to(device)
        target = target.to(device)
        logits = teacher(x)
        probs = F.softmax(logits/train_temp, dim = 1)
        new_labels.append(probs.cpu().detach().numpy())

        torch.cuda.empty_cache()
        del x
        del target
        del logits
        del probs

    new_labels = np.concatenate(new_labels)
    
    teacher_data = TeacherData(mnist_trainset, new_labels)
    teacher_loader = DataLoader(dataset=teacher_data, batch_size=batch_size, shuffle=True)

    print("Start training the student")
    student = train(teacher_loader, file_name, num_epochs, True, train_temp)

    # and finally we predict at temperature 1
    test(student)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mnist_trainset = datasets.MNIST(root='./data/MNIST', train=True, download=True, transform=transforms.ToTensor())
mnist_testset = datasets.MNIST(root='./data/MNIST', train=False, download=True, transform=transforms.ToTensor())

if not os.path.isdir('models'):
    os.makedirs('models')

batch_size = 128
num_epochs = 20

train_loader = DataLoader(dataset=mnist_trainset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=mnist_testset, batch_size=batch_size, shuffle=False)

# Regular training
train(train_loader, "mnist.pt", num_epochs, False)

# Distillation training
train_distillation(train_loader, "mnist_distilled_20.pt", num_epochs, 20)