import os
import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
from Step4_Assemble import DefenseNet


if __name__ == '__main__':

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

    ## training
    model = torch.load('DefenseNet.pth', map_location='cpu')

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


