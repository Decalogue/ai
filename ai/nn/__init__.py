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


def f1_loss(predict, target):
    loss = 0.
    lack_cls = target.sum(dim=0) == 0
    if lack_cls.any():
        loss += F.binary_cross_entropy_with_logits(
            predict[:, lack_cls], target[:, lack_cls])
    predict = torch.sigmoid(predict)
    predict = torch.clamp(predict * (1-target), min=0.01) + predict * target
    tp = predict * target
    tp = tp.sum(dim=0)
    precision = tp / (predict.sum(dim=0) + 1e-8)
    recall = tp / (target.sum(dim=0) + 1e-8)
    f1 = 2 * (precision * recall / (precision + recall + 1e-8))
    return 1. - f1.mean() + loss