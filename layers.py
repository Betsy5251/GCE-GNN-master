import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from aggregator import  GlobalAggregator

def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable

class Conv(nn.Module):
    def __init__(self, in_dim,hop,opt):
        super(Conv, self).__init__()
        self.in_dim = in_dim
        self.hop = hop
        self.sample_num = opt.n_sample



        # self.weight_list = nn.ParameterList(
        #     nn.Parameter(torch.empty(size=(self.in_dim, self.in_dim), dtype=torch.float), requires_grad=True))
        # self.bias_list = nn.ParameterList(
        #     nn.Parameter(torch.empty(size=(1, self.in_dim), dtype=torch.float), requires_grad=True))
        self.linear_1 = nn.Linear(self.in_dim, self.in_dim, bias=True)
        self.linear_2 = nn.Linear(self.in_dim, self.in_dim, bias=True)
        self.linear_3 = nn.Linear(self.in_dim, 1, bias=False)
        self.hiddenSize_pos=opt.hiddenSize_pos
        #global
        self.global_agg = []
        for i in range(self.hop):
            if opt.activate == 'relu':
                agg = GlobalAggregator(self.in_dim, opt.dropout_gcn, self.hiddenSize_pos,act=torch.relu)
            else:
                agg = GlobalAggregator(self.in_dim, opt.dropout_gcn, self.hiddenSize_pos,act=torch.tanh,)
            self.add_module('agg_gcn_{}'.format(i), agg)
            self.global_agg.append(agg)






    def forward(self, h, item_neighbors,embedding,weight_neighbors,item,mask_item,pos_neighbors,pos_before,pos_after):
        batch_size = h.shape[0]
        seqs_len = h.shape[1]
        entity_vectors_ = []
        for j in range(len(item_neighbors)):
            entity_vectors_.append([embedding(i) for i in item_neighbors[j]])  

        entity_vectors_[0][0] = h
        weight_vectors = weight_neighbors
        pos_vectors = pos_neighbors
        session_info = []
        item_emb = nn.embedding(item) * mask_item.float().unsqueeze(-1)

        # mean
        sum_item_emb = torch.sum(item_emb, 1) / torch.sum(mask_item.float(), -1).unsqueeze(-1)

        # sum
        # sum_item_emb = torch.sum(item_emb, 1)

        sum_item_emb = sum_item_emb.unsqueeze(-2)
        for i in range(self.hop):
            session_info.append(sum_item_emb.repeat(1, entity_vectors_[i].shape[1], 1))

        for n_hop in range(self.hop):
            entity_vectors_next_iter = []
            shape = [batch_size, -1, self.sample_num, self.dim]
            for hop in range(self.hop - n_hop):
                aggregator = self.global_agg[n_hop]
                vector = aggregator(self_vectors=entity_vectors_[hop],
                                    neighbor_vector=entity_vectors_[hop+1].view(shape),
                                    masks=None,
                                    batch_size=batch_size,
                                    neighbor_weight=weight_vectors[hop].view(batch_size, -1, self.sample_num),
                                    pos_weight=pos_vectors[hop].view(batch_size, -1, self.sample_num, self.hiddenSize_pos),
                                    extra_vector=session_info[hop])
                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter

        h_global = entity_vectors_[0].view(batch_size, seqs_len, self.dim)
        return h_global






