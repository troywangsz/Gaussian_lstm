from abc import abstractmethod
from math import pi

from torch.distributions import Gamma, Poisson

import torch
import torch.nn as nn
import torch.nn.functional as F
input_size = 7
hidden_size = 16


class GaussianLstm(nn.Module):

    def __init__(self):
        super().__init__()
        self.cell = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=3,
            batch_first=True
        )
        self.linear_m = nn.Linear(
            in_features=hidden_size,
            out_features=1
        )
        self.linear_a = nn.Linear(
            in_features=hidden_size,
            out_features=1
        )

    def forward(self, x):
        o, (_, _) = self.cell(x)
        m, a = self.forward_ma(o)
        return m, a

    def forward_infer(self):
        pass

    def loss(self, label, m, a):
        v = a * a
        t1 = 2 * pi * v
        t1 = torch.pow(t1, -1 / 2)
        t1 = torch.log(t1)

        t2 = label - m
        t2 = torch.pow(t2, 2)
        t2 = - t2
        t2 = t2 / (2 * v)

        loss = t1 + t2

        # loss = torch.exp(loss)
        # loss = torch.log(loss)
        # loss = torch.mean(loss)
        loss = torch.sum(loss)
        loss = -loss

        return loss

    def sample(self, m, a):
        return torch.normal(m, a)

    def forward_ma(self, o):
        m = self.linear_m(o)
        a = F.softplus(self.linear_a(o))
        return m, a
