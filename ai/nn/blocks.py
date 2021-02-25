# -*- coding: utf-8 -*-
""" ai.nn.blocks """
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, maxlen=128):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(maxlen, d_model)
        position = torch.arange(0, maxlen, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


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


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BiLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        self.rnn.flatten_parameters()
        h, _ = self.rnn(input)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        output = self.linear(h)  # batch_size x T x output_size
        return output


class StackedLSTM(nn.Module):
    def __init__(self, num_layers, input_size, hidden_size, dropout):
        super(StackedLSTM, self).__init__()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList()

        for _ in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, hidden_size))
            input_size = hidden_size

    def forward(self, x, hidden):
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(x, (h_0[i], c_0[i]))
            x = h_1_i
            if i + 1 != self.num_layers:
                x = self.dropout(x)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return x, (h_1, c_1)


class StackedGRU(nn.Module):
    def __init__(self, num_layers, input_size, hidden_size, dropout):
        super(StackedGRU, self).__init__()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList()

        for _ in range(num_layers):
            self.layers.append(nn.GRUCell(input_size, hidden_size))
            input_size = hidden_size

    def forward(self, x, hidden):
        h_0 = hidden
        h_1 = []
        for i, layer in enumerate(self.layers):
            h_1_i = layer(x, h_0[i])
            x = h_1_i
            if i + 1 != self.num_layers:
                x = self.dropout(x)
            h_1 += [h_1_i]

        h_1 = torch.stack(h_1)

        return x, h_1