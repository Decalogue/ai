# -*- coding: utf-8 -*-
""" ai.nn """
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
 
    def forward(self, x):
        x = x * F.sigmoid(x)
        return x


class GeM(nn.Module):
    def __init__(self, size=None, p=3, eps=1e-8):
        super(GeM,self).__init__()
        self.size = size
        self.eps = eps
        self.p = Parameter(torch.ones(1) * p)    

    def forward(self, x):
        if self.size is None:
            return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1./self.p)
        else:
            return F.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(self.p), self.size).pow(1./self.p)


class AdaptiveConcatPool2d(nn.Module):
    """ Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`.
    """
    def __init__(self, size=None):
        # Output will be 2*size or 2 if size is None
        super().__init__()
        size = size or 1
        self.ap = nn.AdaptiveAvgPool2d(size)
        self.mp = nn.AdaptiveMaxPool2d(size)

    def forward(self, x):
        return torch.cat([self.ap(x), self.mp(x)], dim=1)