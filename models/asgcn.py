# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .dynamic_rnn import DynamicLSTM
import copy
import numpy as np
#from transformers.modeling_bert import BertLayer,BertLayerNorm,BertcoLayer
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
    def forward(self, text, adj):
        hidden = torch.matmul(text, self.weight)
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        output = torch.matmul(adj.float(), hidden) / denom
        if self.bias is not None:
            return output + self.bias
        else:
            return output
class ASGCN(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(ASGCN, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.text_lstm = DynamicLSTM(opt.input_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.gc1 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        self.gc2 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        self.fc = nn.Linear(2*opt.hidden_dim, opt.polarities_dim)
        self.text_embed_dropout = nn.Dropout(0.3)
    def forward(self, inputs):
        cattext_indices, aspect_indices, left_indices, adj = inputs
        cattext_len = torch.sum(cattext_indices != 0, dim=-1).cpu()
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        left_len = torch.sum(left_indices != 0, dim=-1)
        aspect_double_idx = torch.cat([left_len.unsqueeze(1), (left_len + aspect_len - 1).unsqueeze(1)], dim=1)
        text = self.embed(cattext_indices)
        text = self.text_embed_dropout(text)
        text_out, (_, _) = self.text_lstm(text, cattext_len)
        x = F.relu(self.gc1(self.position_weight(text_out, aspect_double_idx, cattext_len, aspect_len,self.opt.device), adj))
        x = F.relu(self.gc2(self.position_weight(x, aspect_double_idx, cattext_len, aspect_len,self.opt.device), adj))
        x = self.mask(x, aspect_double_idx,self.opt.device)
        alpha_mat = torch.matmul(x, text_out.transpose(1, 2))
        alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)
        x = torch.matmul(alpha, text_out).squeeze(1)  # batch_size x 2*hidden_dim
        output = self.fc(x)
        return output
    def position_weight(self, x, aspect_double_idx, text_len, aspect_len,device):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        text_len = text_len.cpu().numpy()
        aspect_len = aspect_len.cpu().numpy()
        weight = [[] for i in range(batch_size)]
        for i in range(batch_size):
            context_len = text_len[i] - aspect_len[i]
            for j in range(aspect_double_idx[i,0]):
                weight[i].append(1-(aspect_double_idx[i,0]-j)/context_len)
            for j in range(aspect_double_idx[i,0], aspect_double_idx[i,1]+1):
                weight[i].append(0)
            for j in range(aspect_double_idx[i,1]+1, text_len[i]):
                weight[i].append(1-(j-aspect_double_idx[i,1])/context_len)
            for j in range(text_len[i], seq_len):
                weight[i].append(0)
        weight = torch.tensor(weight).unsqueeze(2).float().to(device)
        return weight*x

    def mask(self, x, aspect_double_idx,device):
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
        mask = torch.tensor(mask).unsqueeze(2).float().to(device)
        return mask*x
##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################
class ASBIGCN(nn.Module):
    def __init__(self,embedding_matrix,jiamao):
        super(ASBIGCN, self).__init__()
        self.jiamao=jiamao
        self.embed=nn.Embedding.from_pretrained(torch.tensor(embedding_matrix,dtype=torch.float))
        self.text_embed_dropout = nn.Dropout(0.1)
        self.text_lstm = DynamicLSTM(jiamao.input_dim, jiamao.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.dual_transformer=Dual_Transformer(2*jiamao.hidden_dim)
        nnn=140
        self.nnn=nnn
        self.layer_number=60
        self.fc = nn.Linear(10*jiamao.hidden_dim, 140*500)
        self.lll = nn.ModuleList([DynamicLSTM(nnn, nnn//2,num_layers=3,batch_first=True,bidirectional=True) for _ in range(self.layer_number)])
        self.llll = nn.Linear(140*500, 3)
        self.relu=nn.ReLU()
    def forward(self, inputs):
        span_out,cattext_indices,_, dependency_graph = inputs
        text_len = torch.sum(cattext_indices != 0, dim=-1)

        max_len = cattext_indices.size(1)
        lstm_out,(hout,_)=self.text_lstm(self.text_embed_dropout(self.embed(cattext_indices)),text_len.cpu())
        hout=torch.transpose(hout, 0, 1)
        hout=hout.reshape(hout.size(0),-1)

        text,_ = self.dual_transformer(lstm_out,dependency_graph , length2mask(text_len, max_len,self.jiamao.device))
        x=lstm_out##################
##############################################################################################################################
##############################################################################################################################
        spanlen=max([len(item) for item in span_out])
        tmp=torch.zeros(cattext_indices.size(0),spanlen,2*self.jiamao.hidden_dim).float().to(self.jiamao.device)
        tmp1=torch.zeros(cattext_indices.size(0),spanlen,2*self.jiamao.hidden_dim).float().to(self.jiamao.device)
        for i,spans in enumerate(span_out):
            for j,span in enumerate(spans):
                tmp[i,j]=torch.sum(text[i,span[0]:span[1]],-2)
                tmp1[i,j]=torch.sum(lstm_out[i,span[0]:span[1]],-2)
        v1,_=torch.max(text,-2)
        v2,_=torch.max(x,-2)
        output=self.fc(torch.cat([hout,tmp[:,0,:],tmp1[:,0,:],tmp[:,0,:]*tmp1[:,0,:],torch.abs(tmp[:,0,:]-tmp1[:,0,:])],-1))
        output=torch.reshape(output,(output.shape[0],-1,self.nnn))
        for n in range(self.layer_number):
            output,(_,_)=self.lll[n](output,text_len.cpu())
        output=torch.reshape(output,(output.shape[0],-1))
        output=self.llll(output)

        return output

class Dual_Transformer(nn.Module):
    def __init__(self,in_features):
        super(Dual_Transformer, self).__init__()
        self.in_features = in_features
        self.k = 3
        self.weight = nn.Parameter(torch.FloatTensor(in_features,in_features))
        self.attentions = nn.ModuleList([nn.MultiheadAttention(embed_dim=in_features, num_heads=1, bias=False) for _ in range(self.k)])
        self.dropout=nn.Dropout(0.1)
        self.ffc=nn.Linear(in_features,in_features)
        self.linear = torch.nn.Linear(in_features, in_features, bias=False)
        self.biaffine=Mutual_Biaffine(in_features)
    def forward(self,text,dependency_graph,textmask):
        textmask11 = textmask.unsqueeze(-1)  # b,s1,1
        textmask22 = textmask.unsqueeze(1)  # b,1,s2
        masked = textmask11 * textmask22  # b,s1,s2
        masked=masked>(torch.zeros_like(masked)+0.5)
        out_transformer = text
        out_gcn=text
        for i in range(self.k):
            out_transformer = self.attentions[i](out_transformer,out_transformer,out_transformer, attn_mask=None)[0]+out_transformer###############################有问题
            out_transformer=self.ffc(out_transformer)+out_transformer
            denom1 = torch.sum(dependency_graph, dim=2, keepdim=True) + 0.0000001
            teout = self.linear(out_gcn)

            out_gcn = self.dropout(torch.relu(torch.matmul(dependency_graph.to(torch.float), teout) / denom1)) + out_gcn  ##残差GCN
            out_transformer, out_gcn = self.biaffine(out_transformer, out_gcn, textmask, textmask)
        return out_transformer,out_gcn

class Mutual_Biaffine(nn.Module):
    def __init__(self,in_features):
        super(Mutual_Biaffine, self).__init__()
        self.linear1=torch.nn.Linear(in_features,in_features,bias=False)
        self.linear2=torch.nn.Linear(in_features,in_features,bias=False)
        self.register_parameter('bias', None)############################################ zanshi bu dong
    def forward(self,transformer,gcn,textmask1,textmask2):
        logit1=torch.matmul(self.linear1(transformer),gcn.transpose(1,2))
        logit2=torch.matmul(self.linear1(gcn),transformer.transpose(1,2))
        textmask11 = textmask1.unsqueeze(-1)
        textmask22 = textmask2.unsqueeze(1)
        masked = textmask11 * textmask22
        masked = (1 - masked) * (-10000.0)
        logits1 = torch.softmax(logit1 + masked, -1)   ###经过测试  -1 和-2 结果差不多   更具体一点说的话   -1 貌似好一点
        logits2 = torch.softmax(logit2 + masked, -1)
        output1 = torch.matmul(logits1, gcn)
        output1 = output1 * textmask1.unsqueeze(-1)
        output2 = torch.matmul(logits2, transformer)
        output2 = output2 * textmask2.unsqueeze(-1)  # b,s2,h
        return output1+transformer,output2+gcn

def length2mask(length, maxlength,device):
    size = list(length.size())
    length = length.unsqueeze(-1).repeat(*([1] * len(size) + [maxlength])).long()
    ran = torch.arange(maxlength).to(device)
    ran = ran.expand_as(length)
    mask = ran < length
    return mask.float().to(device)

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])