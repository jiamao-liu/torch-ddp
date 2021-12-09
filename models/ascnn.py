# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .dynamic_rnn import DynamicLSTM

class ASCNN(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(ASCNN, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.text_lstm = DynamicLSTM(opt.input_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.conv1 = nn.Conv1d(2*opt.hidden_dim, 2*opt.hidden_dim, 3, padding=1)  # 3  ===> (3,3)
        self.conv2 = nn.Conv1d(2*opt.hidden_dim, 2*opt.hidden_dim, 3, padding=1)
        self.fc = nn.Linear(2*opt.hidden_dim, opt.polarities_dim)
        self.text_embed_dropout = nn.Dropout(0.3)

    def forward(self, inputs):
        cattext_indices, aspect_indices, left_indices = inputs
        cattext_len = torch.sum(cattext_indices != 0, dim=1).cpu() ###real len
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        left_len = torch.sum(left_indices != 0, dim=-1)
        aspect_double_idx = torch.cat([left_len.unsqueeze(1), (left_len+aspect_len-1).unsqueeze(1)], dim=1)
        text = self.embed(cattext_indices)
        text = self.text_embed_dropout(text)
        text_out, (_, _) = self.text_lstm(text, cattext_len)
        x = F.relu(self.conv1(self.position_weight(text_out, aspect_double_idx, cattext_len, aspect_len).transpose(1,2)))  ### 卷积文字序列必须要转置，因为一个字的向量不同维度才是真正的channel
        x = F.relu(self.conv2(self.position_weight(x.transpose(1,2), aspect_double_idx, cattext_len, aspect_len).transpose(1,2)))
        x = self.mask(x.transpose(1,2), aspect_double_idx)
        alpha_mat = torch.matmul(x, text_out.transpose(1, 2))
        alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)
        x = torch.matmul(alpha, text_out).squeeze(1) # batch_size x 2*hidden_dim
        output = self.fc(x)
        return output

    def position_weight(self, x, aspect_double_idx, cattext_len, aspect_len):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        cattext_len = cattext_len.cpu().numpy()
        aspect_len = aspect_len.cpu().numpy()
        weight = [[] for i in range(batch_size)]
        for i in range(batch_size):
            no_aspect_text_len = cattext_len[i] - aspect_len[i]
            for j in range(aspect_double_idx[i,0]):   ## [i,0] ==   left_len.unsqueeze(1)
                weight[i].append(1-(aspect_double_idx[i,0]-j)/no_aspect_text_len)
            for j in range(aspect_double_idx[i,0], aspect_double_idx[i,1]+1):
                weight[i].append(0)
            for j in range(aspect_double_idx[i,1]+1, cattext_len[i]):
                weight[i].append(1-(j-aspect_double_idx[i,1])/no_aspect_text_len)
            for j in range(cattext_len[i], seq_len):
                weight[i].append(0)########## weight of padding == 0
        weight = torch.tensor(weight).unsqueeze(2).float().cuda()
        return weight*x

    def mask(self, x, aspect_double_idx):
        batch_size, seq_len = x.shape[0], x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        mask = [[] for i in range(batch_size)]
        for i in range(batch_size):
            for j in range(aspect_double_idx[i,0]):
                mask[i].append(0)
            for j in range(aspect_double_idx[i,0], aspect_double_idx[i,1]+1):
                mask[i].append(1)
            for j in range(aspect_double_idx[i,1]+1, seq_len):
                mask[i].append(0)
        mask = torch.tensor(mask).unsqueeze(2).float().cuda()
        return mask*x


