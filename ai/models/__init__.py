# -*- coding: utf-8 -*-
""" ai.models """
import torch
import torch.nn as nn
import torch.nn.functional as F


def match_score(seq1, seq2, mask1, mask2):
    """
    seq1, seq2: batch_size * seq_len * emb_dim
    """
    batch, seq_len, emb_dim = seq1.shape
    seq1 = seq1 * mask1.eq(0).unsqueeze(2).float()
    seq2 = seq2 * mask2.eq(0).unsqueeze(2).float()
    seq1 = seq1.unsqueeze(2).repeat(1, 1, seq_len, 1)
    seq2 = seq2.unsqueeze(1).repeat(1, seq_len, 1, 1)
    a = seq1 - seq2
    a = torch.norm(a, dim=-1, p=2)
    return 1.0 / (1.0 + a)


def attention_avg_pooling(seq1, seq2, mask1, mask2):
    """
    A: batch_size * seq_len * seq_len
    """
    A = match_score(seq1, seq2, mask1, mask2)
    weight1 = torch.sum(A, -1)
    weight2 = torch.sum(A.transpose(1, 2), -1)
    s1 = seq1 * weight1.unsqueeze(2)
    s2 = seq2 * weight2.unsqueeze(2)
    s1 = F.avg_pool1d(s1.transpose(1, 2), kernel_size=3, padding=1, stride=1)
    s2 = F.avg_pool1d(s2.transpose(1, 2), kernel_size=3, padding=1, stride=1)
    s1, s2 = s1.transpose(1, 2), s2.transpose(1, 2)
    return s1, s2


class WideConv(nn.Module):
    def __init__(self, seq_len, emb_dim, device="gpu"):
        super(WideConv, self).__init__()
        self.seq_len = seq_len
        self.emb_dim = emb_dim
        self.W = nn.Parameter(torch.randn((seq_len, emb_dim)))
        nn.init.xavier_normal_(self.W)
        self.W.to(device)
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, padding=[1, 1], stride=1)
        self.tanh = nn.Tanh()

    def forward(self, seq1, seq2, mask1, mask2):
        """
        seq1, seq2: batch_size * seq_len * emb_dim
        A: batch_size * seq_len * seq_len
        x: batch_size * 2 * seq_len * emb_dim
        """
        A = match_score(seq1, seq2, mask1, mask2)
        # attn_feature_map: batch_size * seq_len * emb_dim
        attn_feature_map1 = A.matmul(self.W)
        attn_feature_map2 = A.transpose(1, 2).matmul(self.W)
        x1 = torch.cat([seq1.unsqueeze(1), attn_feature_map1.unsqueeze(1)], 1)
        x2 = torch.cat([seq2.unsqueeze(1), attn_feature_map2.unsqueeze(1)], 1)
        o1, o2 = self.conv(x1).squeeze(1), self.conv(x2).squeeze(1)
        o1, o2 = self.tanh(o1), self.tanh(o2)
        return o1, o2


class ABCNN(nn.Module):
    def __init__(self, embeddings, num_layer=1, hidden_size=300, maxlen=64, device="gpu", finetune=True):
        super(ABCNN, self).__init__()
        self.num_layer = num_layer
        self.hidden_size = hidden_size
        self.device = device

        self.emb_dim = embeddings.shape[1]
        self.emb = nn.Embedding(embeddings.shape[0], embeddings.shape[1])
        self.emb.weight = nn.Parameter(torch.from_numpy(embeddings))
        self.emb.float()
        if finetune:
            self.emb.weight.requires_grad = True
        self.emb.to(device)

        self.conv = nn.ModuleList([WideConv(maxlen, self.emb_dim, device) for _ in range(self.num_layer)])
        self.fc = nn.Sequential(
            nn.Linear(self.emb_dim * (1+self.num_layer) * 2, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, 2)
        )

    def forward(self, seq1, seq2):
        res = [[], []]
        mask1, mask2 = seq1.eq(0), seq2.eq(0)
        seq1_encode = self.emb(seq1)
        seq2_encode = self.emb(seq2)
        # (batch_size, seq_len, emb_dim) => (batch_size, emb_dim)
        res[0].append(F.avg_pool1d(seq1_encode.transpose(1, 2), kernel_size=seq1_encode.size(1)).squeeze(-1))
        res[1].append(F.avg_pool1d(seq2_encode.transpose(1, 2), kernel_size=seq2_encode.size(1)).squeeze(-1))
        for i, conv in enumerate(self.conv):
            o1, o2 = conv(seq1_encode, seq2_encode, mask1, mask2)
            res[0].append(F.avg_pool1d(o1.transpose(1, 2), kernel_size=o1.size(1)).squeeze(-1))
            res[1].append(F.avg_pool1d(o2.transpose(1, 2), kernel_size=o2.size(1)).squeeze(-1))
            o1, o2 = attention_avg_pooling(o1, o2, mask1, mask2)
            seq1_encode, seq2_encode = o1 + seq1_encode, o2 + seq2_encode
        # batch_size * (emb_dim * (1+num_layer) * 2) => batch_size * hidden_size
        x = torch.cat([torch.cat(res[0], 1), torch.cat(res[1], 1)], 1)
        out = self.fc(x)
        prob = F.softmax(out, dim=-1)
        return out, prob


class SiaRNN(nn.Module):
    def __init__(self, embeddings, num_layer=2, hidden_size=300, maxlen=64, device="gpu", finetune=True):
        super(SiaRNN, self).__init__()
        self.device = device
        self.num_layer = num_layer
        self.hidden_size = hidden_size

        self.emb_dim = embeddings.shape[1]
        self.emb = nn.Embedding(embeddings.shape[0], embeddings.shape[1])
        self.emb.weight = nn.Parameter(torch.from_numpy(embeddings))
        self.emb.float()
        if finetune:
            self.emb.weight.requires_grad = True
        self.emb.to(device)
        
        self.rnn = nn.LSTM(self.emb_dim, self.hidden_size, batch_first=True, bidirectional=True, num_layers=self.num_layer)
        self.fc = nn.Linear(maxlen, 2)
    
    def dropout(self, x):
        return F.dropout(x, p=0.2, training=self.training)

    def forward(self, seq1, seq2):
        # emb: batch_size * seq_len => batch_size * seq_len * emb_dim
        seq1_encode = self.emb(seq1)
        seq2_encode = self.emb(seq2)
        seq1_encode = self.dropout(seq1_encode)
        seq2_encode = self.dropout(seq2_encode)
        
        output1, _ = self.rnn(seq1_encode)
        output2, _ = self.rnn(seq2_encode)
        x = torch.exp(-torch.norm(output1 - output2, p=2, dim=-1, keepdim=True))
        out = self.fc(x.squeeze(dim=-1))
        prob = F.softmax(out, dim=-1)
        return out, prob