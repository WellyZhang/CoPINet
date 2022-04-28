# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


def conv1x1(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel,
                     out_channel,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


def conv3x3(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel,
                     out_channel,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class MLP(nn.Module):

    def __init__(self, in_dim=256, out_dim=8, dropout=False):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, in_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(in_dim, out_dim)
        if dropout:
            self.dropout = nn.Dropout(0.5)
        else:
            self.dropout = Identity()

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class ResBlock(nn.Module):

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3(in_channel, out_channel, stride)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channel, out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        if downsample is None:
            self.downsample = Identity()
        else:
            self.downsample = downsample

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.downsample(x) + self.bn2(self.conv2(out)))

        return out


class GumbelSoftmax(nn.Module):

    def __init__(self, interval=100, temperature=1.0):
        super(GumbelSoftmax, self).__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=-1)
        self.anneal_rate = 0.00003
        self.interval = 100
        self.counter = 0
        self.temperature_min = 0.5

    def anneal(self):
        self.temperature = max(
            self.temperature * torch.exp(-self.anneal_rate * self.counter),
            self.temperature_min)

    def sample_gumbel(self, logits, eps=1e-20):
        U = torch.rand_like(logits)
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits):
        y = logits + self.sample_gumbel(logits)
        return self.softmax(y / self.temperature)

    def forward(self, logits):
        self.counter += 1
        if self.counter % self.interval == 0:
            self.anneal()
        y = self.gumbel_softmax_sample(logits)
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y)
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = (y_hard - y).detach() + y
        return y_hard
