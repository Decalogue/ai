# -*- coding: utf-8 -*-
""" ai.loss """
import torch
import torch.nn as nn
import torch.nn.functional as F


class F1Loss(nn.Module):
    def __init__(self, eps=1e-8):
        super(F1Loss, self).__init__()
        self.eps = eps

    def forward(self, x, target):
        loss = 0.
        lack_cls = target.sum(dim=0) == 0
        if lack_cls.any():
            loss += F.binary_cross_entropy_with_logits(
                x[:, lack_cls], target[:, lack_cls])
        x = torch.sigmoid(x)
        x = torch.clamp(x * (1-target), min=0.01) + x * target
        tp = x * target
        tp = tp.sum(dim=0)
        precision = tp / (x.sum(dim=0) + self.eps)
        recall = tp / (target.sum(dim=0) + self.eps)
        f1 = 2 * (precision * recall / (precision + recall + self.eps))
        return 1. - f1.mean() + loss


class LabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothing, self).__init__()
        self.smoothing = smoothing
    
    def forward(self, x, target):
        if self.training:
            log_prob = F.log_softmax(x, dim=-1)
            weight = x.new_ones(x.size()) * self.smoothing / (x.size(-1) - 1)
            weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
            loss = (-weight * log_prob).sum(dim=-1).mean()
        else:
            loss = F.cross_entropy(x, target)
        return loss


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class SoftTargetCrossEntropy(nn.Module):
    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x, target):
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()