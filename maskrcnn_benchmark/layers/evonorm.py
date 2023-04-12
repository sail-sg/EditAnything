import torch
import torch.nn as nn


class EvoNorm2d(nn.Module):
    __constants__ = ['num_features', 'eps', 'nonlinearity']

    def __init__(self, num_features, eps=1e-5, nonlinearity=True, group=32):
        super(EvoNorm2d, self).__init__()

        self.num_features = num_features
        self.eps = eps
        self.nonlinearity = nonlinearity
        self.group = group

        self.weight = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        if self.nonlinearity:
            self.v = nn.Parameter(torch.Tensor(1, num_features, 1, 1))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)
        if self.nonlinearity:
            nn.init.ones_(self.v)

    def group_std(self, x, groups=32):
        N, C, H, W = x.shape
        x = torch.reshape(x, (N, groups, C // groups, H, W))
        std = torch.std(x, (3, 4), keepdim=True)
        return torch.reshape(std + self.eps, (N, C, 1, 1))

    def forward(self, x):
        if self.nonlinearity:
            num = x * torch.sigmoid(self.v * x)
            return num / self.group_std(x, self.group) * self.weight + self.bias
        else:
            return x * self.weight + self.bias