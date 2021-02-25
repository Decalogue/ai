# -*- coding: utf-8 -*-
""" ai.nn.attention """
import torch
import torch.nn as nn
import torch.nn.functional as F


class maxout(nn.Module):
    def __init__(self, in_feature, out_feature, pool_size):
        super(maxout, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.pool_size = pool_size
        self.linear = nn.Linear(in_feature, out_feature * pool_size)

    def forward(self, x):
        output = self.linear(x)
        output = output.view(-1, self.out_feature, self.pool_size)
        output = output.max(2)[0]
        return output


class BahdanauAttention(nn.Module):
    """ Implementation of the Bahdanau Attention.
        https://arxiv.org/pdf/1409.0473
    """
    def __init__(self, hidden_size, emb_size):
        super(BahdanauAttention, self).__init__()
        self.hidden_size = hidden_size
        self.emb_size = emb_size
        self.linear_encoder = nn.Linear(hidden_size, hidden_size)
        self.linear_decoder = nn.Linear(hidden_size, hidden_size)
        self.linear_v = nn.Linear(hidden_size, 1)
        self.linear_r = nn.Linear(hidden_size * 2 + emb_size, hidden_size * 2)
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()

    def init_context(self, context):
        self.context = context.transpose(0, 1)

    def forward(self, h, x):
        gamma_encoder = self.linear_encoder(self.context)      # batch * time * size
        gamma_decoder = self.linear_decoder(h).unsqueeze(1)    # batch * 1 * size
        weights = self.linear_v(self.tanh(gamma_encoder + gamma_decoder)).squeeze(2)   # batch * time
        weights = self.softmax(weights)   # batch * time
        c_t = torch.bmm(weights.unsqueeze(1), self.context).squeeze(1)  # batch * size
        r_t = self.linear_r(torch.cat([c_t, h, x], dim=1))
        output = r_t.view(-1, self.hidden_size, 2).max(2)[0]

        return output, weights


class LuongAttention(nn.Module):
    """ Implementation of the Luong Attention.
        https://arxiv.org/pdf/1508.04025
    """
    def __init__(self, hidden_size, emb_size, pool_size=0):
        super(LuongAttention, self).__init__()
        self.hidden_size, self.emb_size, self.pool_size = hidden_size, emb_size, pool_size
        self.linear_in = nn.Linear(hidden_size, hidden_size)
        if pool_size > 0:
            self.linear_out = maxout(2*hidden_size + emb_size, hidden_size, pool_size)
        else:
            self.linear_out = nn.Sequential(nn.Linear(2*hidden_size + emb_size, hidden_size), nn.SELU(), 
                                            nn.Linear(hidden_size, hidden_size), nn.Tanh())
        self.softmax = nn.Softmax(dim=1)

    def init_context(self, context):
        self.context = context.transpose(0, 1)

    def forward(self, h, x):
        gamma_h = self.linear_in(h).unsqueeze(2)    # batch * size * 1
        weights = torch.bmm(self.context, gamma_h).squeeze(2)   # batch * time
        weights = self.softmax(weights)   # batch * time
        c_t = torch.bmm(weights.unsqueeze(1), self.context).squeeze(1) # batch * size
        output = self.linear_out(torch.cat([c_t, h, x], dim=1))

        return output, weights


class LuongGateAttention(nn.Module):
    """ Implementation of the Luong Attention.
    """
    def __init__(self, hidden_size, emb_size, prob=0.1):
        super(LuongGateAttention, self).__init__()
        self.linear_in = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.SELU(), nn.Dropout(p=prob),
                                       nn.Linear(hidden_size, hidden_size), nn.SELU(), nn.Dropout(p=prob))
        self.linear_out = nn.Sequential(nn.Linear(2*hidden_size, hidden_size), nn.SELU(), nn.Dropout(p=prob),
                                        nn.Linear(hidden_size, hidden_size), nn.SELU(), nn.Dropout(p=prob))
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=prob)

    def init_context(self, context):
        self.context = context.transpose(0, 1)

    def forward(self, h):
        gamma_h = self.linear_in(h).unsqueeze(2)
        weights = self.dropout(torch.bmm(self.context, gamma_h).squeeze(2))
        weights = self.softmax(weights)
        c_t = torch.bmm(weights.unsqueeze(1), self.context).squeeze(1)
        output = self.linear_out(torch.cat([h, c_t], dim=1))

        return output, weights
