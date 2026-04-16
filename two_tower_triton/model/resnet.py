"""ResNet building block for Two Tower model.

Ported from GemmKernelSelection/Embedding/two_tower/model/resnet.py.
Used as an optional residual backbone in QueryEncoder / DocumentEncoder.
"""

from itertools import pairwise

from torch import nn


class ResBlock(nn.Module):
    def __init__(self, in_size, out_size, dropout_rate=0):
        super().__init__()

        self.block = nn.Sequential(
            nn.BatchNorm1d(in_size),
            nn.ReLU(),
            nn.Linear(in_size, out_size),
            nn.BatchNorm1d(out_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(out_size, out_size),
        )

        self.shortcut = None if in_size == out_size else nn.Linear(in_size, out_size)

    def forward(self, x):
        residual = x
        out = self.block(x)
        if self.shortcut:
            residual = self.shortcut(residual)
        return out + residual


class ResNet(nn.Module):
    def __init__(self, in_size, hidden_sizes, dropout_rate=0):
        super().__init__()

        self.lin = nn.Linear(in_size, hidden_sizes[0])
        self.net = nn.Sequential()
        for in_s, out_s in pairwise(hidden_sizes):
            self.net.append(ResBlock(in_s, out_s, dropout_rate))
        self.bn = nn.BatchNorm1d(hidden_sizes[-1])
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.lin(x)
        out = self.net(out)
        out = self.bn(out)
        out = self.relu(out)
        return out
