import torch
from torch import nn
from torch.autograd import Variable
from collections import Counter
import numpy as np
m1 = nn.Threshold(0.5, 0)
m2 = nn.Threshold(-0.5, 1)
input = Variable(torch.randn([5,5]), requires_grad = True)
output = m1(input)
output = -output
output = m2(output)
output = output.detach().view(-1).numpy()
print(input)
print(output)
count1 = Counter()
print(count1)
count2 = Counter(output)
print(count2)
count3 = count1 + count2
print(count3)
print(sum(count3.values()))


