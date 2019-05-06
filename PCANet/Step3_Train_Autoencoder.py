import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            # nn.Linear(200, 128),
            # nn.Tanh(),
            # nn.Linear(128, 64),
            # nn.Tanh(),
            # nn.Linear(64, 32),
            # nn.Tanh(),
            nn.Linear(200, 10))
        self.decoder = nn.Sequential(
            # nn.Linear(32, 32),
            # nn.Tanh(),
            # nn.Linear(32, 64),
            # nn.Tanh(),
            # nn.Linear(64, 128),
            # nn.Tanh(),
            nn.Linear(10, 200),
            nn.ReLU())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

if __name__ == '__main__':


    data = np.load('res.npy')
    devdata = np.load('res_dev.npy')
    use_cuda = torch.cuda.is_available()
    num_epochs = 10000
    batch_size = 128
    learning_rate = 1e-3

    dataset = data, data
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    dataset_dev = devdata, devdata
    dataloader_dev = DataLoader(dataset_dev, batch_size=batch_size, shuffle=True)

    model = autoencoder()
    if use_cuda:
        model = model.cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    bestloss = 1e-4
    for epoch in range(num_epochs):
        model.train()
        for data in dataloader:
            optimizer.zero_grad()
            features, _ = data
            if use_cuda:
                features = Variable(features).cuda()
            # ===================forward=====================
            output = model(features)
            loss = criterion(output, features)
            tooutput = output[0].detach().cpu().numpy()
            tofeatures = features[0].detach().cpu().numpy()
            # ===================backward====================
            loss.backward()
            optimizer.step()
        # ===================log========================
        # ------------------------- ------------------------- -------------------------
        # ------------------------- ------------------------- -------------------------
        # ------------------------- ------------------------- -------------------------
        # model.eval()
        # for data in dataloader_dev:
        #     features, _ = data
        #     if use_cuda:
        #         features = Variable(features).cuda()
        #     # ===================forward=====================
        #     output = model(features)
        #     loss2 = criterion(output, features)
        #     tooutput = output[0].detach().cpu().numpy()
        #     tofeatures = features[0].detach().cpu().numpy()
        # ===================log========================

        print('epoch [{}/{}], train loss:{:.4f}, '
              .format(epoch + 1, num_epochs, loss.data.item()),
              'distance: ', np.linalg.norm(tofeatures - tooutput))
        print('---------')


    torch.save(model, 'AEC.pth')
