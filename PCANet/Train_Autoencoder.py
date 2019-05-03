import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})


data = np.load('res.npy')
use_cuda = torch.cuda.is_available()
num_epochs = 10000
batch_size = 128
learning_rate = 1e-3

dataset = data, data
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


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


model = autoencoder()
if use_cuda:
    model = model.cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
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
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch + 1, num_epochs, loss.data.item()))
    # print(tofeatures[:10])
    # print(tooutput[:10])
    print('distance: ', np.linalg.norm(tofeatures-tooutput))
    print('---------')


torch.save(model, 'AEC.pth')
