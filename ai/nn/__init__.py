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