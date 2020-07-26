# -*- coding: utf-8 -*-
""" ai.nn.blocks """
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class ResnetBlock(nn.Module):
    def __init__(self, dim, use_bias=False):
        super(ResnetBlock, self).__init__()
        conv_block = []
        conv_block += [nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias),
                       nn.InstanceNorm2d(dim),
                       nn.ReLU(True)]

        conv_block += [nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias),
                       nn.InstanceNorm2d(dim)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class LIN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(LIN, self).__init__()
        self.eps = eps
        self.rho = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.gamma = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.beta = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.rho.data.fill_(0.0)
        self.gamma.data.fill_(1.0)
        self.beta.data.fill_(0.0)

    def forward(self, input):
        in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True), torch.var(input, dim=[1, 2, 3], keepdim=True)
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
        out = self.rho.expand(input.shape[0], -1, -1, -1) * out_in + (1-self.rho.expand(input.shape[0], -1, -1, -1)) * out_ln
        out = out * self.gamma.expand(input.shape[0], -1, -1, -1) + self.beta.expand(input.shape[0], -1, -1, -1)
        return out


class adaLIN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(adaLIN, self).__init__()
        self.eps = eps
        self.rho = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.rho.data.fill_(0.9)

    def forward(self, input, gamma, beta):
        in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True), torch.var(input, dim=[1, 2, 3], keepdim=True)
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
        out = self.rho.expand(input.shape[0], -1, -1, -1) * out_in + (1-self.rho.expand(input.shape[0], -1, -1, -1)) * out_ln
        out = out * gamma.unsqueeze(2).unsqueeze(3) + beta.unsqueeze(2).unsqueeze(3)
        return out


class SoftAdaLIN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(SoftAdaLIN, self).__init__()
        self.norm = adaLIN(num_features, eps)

        self.w_gamma = Parameter(torch.zeros(1, num_features))
        self.w_beta = Parameter(torch.zeros(1, num_features))

        self.c_gamma = nn.Sequential(nn.Linear(num_features, num_features),
                                     nn.ReLU(True),
                                     nn.Linear(num_features, num_features))
        self.c_beta = nn.Sequential(nn.Linear(num_features, num_features),
                                    nn.ReLU(True),
                                    nn.Linear(num_features, num_features))
        self.s_gamma = nn.Linear(num_features, num_features)
        self.s_beta = nn.Linear(num_features, num_features)

    def forward(self, x, content_features, style_features):
        content_gamma, content_beta = self.c_gamma(content_features), self.c_beta(content_features)
        style_gamma, style_beta = self.s_gamma(style_features), self.s_beta(style_features)

        w_gamma, w_beta = self.w_gamma.expand(x.shape[0], -1), self.w_beta.expand(x.shape[0], -1)
        soft_gamma = (1. - w_gamma) * style_gamma + w_gamma * content_gamma
        soft_beta = (1. - w_beta) * style_beta + w_beta * content_beta

        out = self.norm(x, soft_gamma, soft_beta)
        return out


class ResnetSoftAdaLINBlock(nn.Module):
    def __init__(self, dim, use_bias=False):
        super(ResnetSoftAdaLINBlock, self).__init__()
        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias)
        self.norm1 = SoftAdaLIN(dim)
        self.relu1 = nn.ReLU(True)

        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias)
        self.norm2 = SoftAdaLIN(dim)

    def forward(self, x, content_features, style_features):
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.norm1(out, content_features, style_features)
        out = self.relu1(out)

        out = self.pad2(out)
        out = self.conv2(out)
        out = self.norm2(out, content_features, style_features)
        return out + x


class ResnetAdaLINBlock(nn.Module):
    def __init__(self, dim, use_bias=False):
        super(ResnetAdaLINBlock, self).__init__()
        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias)
        self.norm1 = adaLIN(dim)
        self.relu1 = nn.ReLU(True)

        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias)
        self.norm2 = adaLIN(dim)

    def forward(self, x, gamma, beta):
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.norm1(out, gamma, beta)
        out = self.relu1(out)
        out = self.pad2(out)
        out = self.conv2(out)
        out = self.norm2(out, gamma, beta)
        return out + x


class RhoClipper(object):
    def __init__(self, min, max):
        self.clip_min = min
        self.clip_max = max
        assert min < max

    def __call__(self, module):
        if hasattr(module, 'rho'):
            w = module.rho.data
            w = w.clamp(self.clip_min, self.clip_max)
            module.rho.data = w


class WClipper(object):
    def __init__(self, min, max):
        self.clip_min = min
        self.clip_max = max
        assert min < max

    def __call__(self, module):
        if hasattr(module, 'w_gamma'):
            w = module.w_gamma.data
            w = w.clamp(self.clip_min, self.clip_max)
            module.w_gamma.data = w

        if hasattr(module, 'w_beta'):
            w = module.w_beta.data
            w = w.clamp(self.clip_min, self.clip_max)
            module.w_beta.data = w