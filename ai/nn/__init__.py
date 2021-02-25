# -*- coding: utf-8 -*-
""" ai.nn """
import torch
import torch.nn as nn
import torch.nn.functional as F


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
 
    def forward(self, x):
        x = x * F.sigmoid(x)
        return x


def f1_loss(predict, target, eps=1e-8):
    loss = 0.
    lack_cls = target.sum(dim=0) == 0
    if lack_cls.any():
        loss += F.binary_cross_entropy_with_logits(
            predict[:, lack_cls], target[:, lack_cls])
    predict = torch.sigmoid(predict)
    predict = torch.clamp(predict * (1-target), min=0.01) + predict * target
    tp = predict * target
    tp = tp.sum(dim=0)
    precision = tp / (predict.sum(dim=0) + eps)
    recall = tp / (target.sum(dim=0) + eps)
    f1 = 2 * (precision * recall / (precision + recall + eps))
    return 1. - f1.mean() + loss


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