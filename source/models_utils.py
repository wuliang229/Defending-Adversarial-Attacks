import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.autograd.function import Function


class ANL(nn.Module):
    def __init__(self):
        super(ANL, self).__init__()

    def forward(self, x, factor=0.02):
        if factor is not None:
            gaussian = x.new([Normal(factor / 2, factor / 4).sample()]).clamp(0, factor)
            std = torch.std(x)
            grad = x.grad
            noise = gaussian * std * grad / torch.norm(grad, float('inf'), keepdim=True)
        else:
            noise = 0
        return x + noise


class BackBlock(Function):
    def __init__(self):
        super(BackBlock, self).__init__()

    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = input.new(torch.zeros(size=input.shape))
        if ctx.needs_input_grad[1]:
            grad_weight = weight.new(torch.zeros(size=weight.shape))
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = bias.new(torch.zeros(size=bias.shape))

        return grad_input, grad_weight, grad_bias
