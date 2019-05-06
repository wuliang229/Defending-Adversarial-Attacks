import torch
from torch import nn
from torch.autograd import Variable
m = nn.Threshold(0.1, 20)
input = Variable(torch.randn([2]), requires_grad = True)
output = m(input)
print(input)
print(output)
output.backward()
loss = nn.CrossEntropyLoss(input, output)
loss.backward()


