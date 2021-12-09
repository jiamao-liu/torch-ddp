
# -*- coding: utf-8 -*-


from .dynamic_rnn import DynamicLSTM
import torch
import torch.nn as nn


class LSTM(nn.Module):

    def __init__(self, embedding_matrix, opt):
        super(LSTM, self).__init__()
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.lstm = DynamicLSTM(opt.input_dim, opt.hidden_dim, num_layers=1, batch_first=True,dropout=opt.drop_out)
        self.fc = nn.Linear(opt.hidden_dim, opt.polarities_dim)

    def forward(self, inputs):
        text_indices = inputs[0]
        x = self.embed(text_indices)
        x_len = torch.sum(text_indices != 0, dim=-1).cpu()
        _, (h_n, _) = self.lstm(x, x_len)
        out = self.fc(h_n[0])
        return out