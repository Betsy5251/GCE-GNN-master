import datetime
import math
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from aggregator import LocalAggregator, GlobalAggregator
from torch.nn import Module, Parameter
import torch.nn.functional as F
from GlobalConv_D import Globalgarph_D

class CombineGraph(Module):
    def __init__(self, opt, num_node, adj_all, num, pos_all):
        super(CombineGraph, self).__init__()
        self.opt = opt

        self.batch_size = opt.batch_size
        self.num_node = num_node
        self.dim = opt.hiddenSize
        self.dropout_local = opt.dropout_local
        self.dropout_global = opt.dropout_global
        self.hop = opt.n_iter
        self.sample_num = opt.n_sample
        self.adj_all = trans_to_cuda(torch.Tensor(adj_all)).long()
        self.num = trans_to_cuda(torch.Tensor(num)).float()
        self.pos_all = trans_to_cuda(torch.Tensor(pos_all)).float()
        self.hiddenSize_pos = opt.hiddenSize_pos
        self.beta = opt.conbeta
        # self.beta = opt.conbeta
        self.D_global_agg = Globalgarph_D(self.hop, opt)
        # Aggregator
        self.local_agg = LocalAggregator(self.dim, self.opt.alpha, dropout=0.0)
        self.global_agg = []
        for i in range(self.hop):
            if opt.activate == 'relu':
                agg = GlobalAggregator(self.dim, opt.dropout_gcn, self.hiddenSize_pos, act=torch.relu)
            else:
                agg = GlobalAggregator(self.dim, opt.dropout_gcn, self.hiddenSize_pos, act=torch.tanh)
            self.add_module('agg_gcn_{}'.format(i), agg)
            self.global_agg.append(agg)

        # Item representation & Position representation
        self.embedding = nn.Embedding(num_node, self.dim)
        self.pos_embedding = nn.Embedding(300, self.dim) # 200
        # pos
        self.pos_before = nn.Embedding(10, opt.hiddenSize_pos)
        self.pos_after = nn.Embedding(10, opt.hiddenSize_pos)
        self.pos_io = nn.Embedding(2, opt.hiddenSize_pos)
        # Parameters
        self.w_1 = nn.Parameter(torch.Tensor(2 * self.dim, self.dim))
        self.w_2 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.glu1 = nn.Linear(self.dim, self.dim)
        self.glu2 = nn.Linear(self.dim, self.dim, bias=False)
        self.linear_transform = nn.Linear(self.dim, self.dim, bias=False)

        self.leakyrelu = nn.LeakyReLU(opt.alpha)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def sample(self, target): # def sample(self, target, n_sample):
        # neighbor = self.adj_all[target.view(-1)]
        # index = np.arange(neighbor.shape[1])
        # np.random.shuffle(index)
        # index = index[:n_sample]
        # return self.adj_all[target.view(-1)][:, index], self.num[target.view(-1)][:, index]
        # return self.adj_all[target.view(-1)], self.num[target.view(-1)]
        adj_all = []
        num_all = []
        pos_all = []
        b = len(target)
        for j in range(len(target)):
            adj_in = self.adj_all[0][target[j].view(-1)]
            num_in = self.num[0][target[j].view(-1)]
            pos_in = self.pos_all[0][target[j].view(-1)]
            adj_out = self.adj_all[1][target[j].view(-1)]
            num_out = self.num[1][target[j].view(-1)]
            pos_out = self.pos_all[1][target[j].view(-1)]
            adj_io = self.adj_all[2][target[j].view(-1)]
            num_io = self.num[2][target[j].view(-1)]
            adj_all.append(adj_in)
            adj_all.append(adj_out)
            adj_all.append(adj_io)
            num_all.append(num_in)
            num_all.append(num_out)
            num_all.append(num_io)
            pos_all.append(pos_in)
            pos_all.append(pos_out)
            pos_all.append(pos_out)
        return adj_all, num_all, pos_all

    def SSL(self, sess_emb_hgnn, sess_emb_lgcn):

        def row_column_shuffle(embedding):
            corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
            corrupted_embedding = corrupted_embedding[:,torch.randperm(corrupted_embedding.size()[1])]
            return corrupted_embedding
        def score(x1, x2):
            return torch.sum(torch.mul(x1, x2), 1)

        pos = trans_to_cuda(score(sess_emb_hgnn, sess_emb_lgcn))
        neg1 = trans_to_cuda(score(sess_emb_lgcn, row_column_shuffle(sess_emb_hgnn)))
        one  = trans_to_cuda(torch.ones(neg1.shape[0]))
        con_loss = trans_to_cuda(torch.sum(-torch.log(1e-8 + torch.sigmoid(pos))-torch.log(1e-8 + (one - torch.sigmoid(neg1)))))
        return self.beta*con_loss

    def compute_scores(self, hidden,global_hidden, mask):
        # local
        hidden = F.dropout(hidden,self.dropout_local,training=self.training)
        mask = mask.float().unsqueeze(-1)
        batch_size = hidden.shape[0]
        len = hidden.shape[1]
        pos_emb = self.pos_embedding.weight[:len]
        pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)

        hs = torch.sum(hidden * mask, -2) / torch.sum(mask, 1)
        hs = hs.unsqueeze(-2).repeat(1, len, 1)
        nh = torch.matmul(torch.cat([pos_emb, hidden], -1), self.w_1)
        nh = torch.tanh(nh)
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))
        beta = torch.matmul(nh, self.w_2)
        beta = beta * mask
        select = torch.sum(beta * hidden, 1)

        # global
        global_hidden = F.dropout(global_hidden, self.dropout_global, training=self.training)
        pos_emb = self.pos_embedding.weight[:len]
        pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)

        hs_global = torch.sum(global_hidden * mask, -2) / torch.sum(mask, 1)
        hs_global = hs_global.unsqueeze(-2).repeat(1, len, 1)
        nh_global = torch.matmul(torch.cat([pos_emb, global_hidden], -1), self.w_1)
        nh_global = torch.tanh(nh_global)
        nh_global = torch.sigmoid(self.glu1(nh_global) + self.glu2(hs_global))
        beta_global = torch.matmul(nh_global, self.w_2)
        beta_global = beta_global * mask
        h_global = torch.sum(beta_global * global_hidden,1)

        b = self.embedding.weight[1:]  # n_nodes x latent_size
        conloss = self.SSL(select,h_global)
        output = select + h_global
        scores = torch.matmul(output, b.transpose(1, 0))
        return scores, conloss

    def forward(self, inputs, adj, mask_item, item):
        batch_size = inputs.shape[0]
        seqs_len = inputs.shape[1]
        h = self.embedding(inputs)

        # local
        h_local = self.local_agg(h, adj, mask_item)

        # global
        item_neighbors = [[inputs]]
        weight_neighbors = []
        pos_neighbors = []
        support_size = seqs_len

        for i in range(1, self.hop + 1):
            item_sample_i, weight_sample_i, pos_sample_i = self.sample(item_neighbors[i - 1])
            support_size *= self.sample_num
            item_neighbors.append([item_sample_i[j].view(batch_size, support_size) for j in range(len(item_sample_i))])
            weight_neighbors.append(
                [weight_sample_i[j].view(batch_size, support_size) for j in range(len(weight_sample_i))])
            pos_neighbors.append([pos_sample_i[j].view(batch_size, support_size) for j in range(len(pos_sample_i))])

        entity_vectors_ = []
        for j in range(len(item_neighbors)):
            entity_vectors_.append([self.embedding(i) for i in item_neighbors[j]])

        weight_vectors = weight_neighbors
        pos_vectors = pos_neighbors
        session_info = []
        item_emb = self.embedding(item) * mask_item.float().unsqueeze(-1)
        hh = [self.embedding(i) for i in item_neighbors[0]][0]
        for j in range(len(pos_neighbors)):
            for i in range(0,len(pos_neighbors[j]),3):
                pos_neighbors[j][i]= self.pos_before(pos_neighbors[j][i].long())

            for k in range(1,len(pos_neighbors[j]),3):
                pos_neighbors[j][k] = self.pos_after(pos_neighbors[j][k].long())

            for z in range(2,len(pos_neighbors[j]),3):
                pos_neighbors[j][z] = self.pos_io(trans_to_cuda(torch.tensor([1]))).unsqueeze(0).repeat(pos_neighbors[j][k].shape[0],pos_neighbors[j][k].shape[1],1)

        # mean
        sum_item_emb = torch.sum(item_emb, 1) / torch.sum(mask_item.float(), -1).unsqueeze(-1)

        # sum
        # sum_item_emb = torch.sum(item_emb, 1)

        sum_item_emb = sum_item_emb.unsqueeze(-2)
        for j in range(len(entity_vectors_)):
            for i in range(self.hop):
                session_info.append(sum_item_emb.repeat(1, entity_vectors_[j][i].shape[1], 1))

        for n_hop in range(self.hop):
            entity_vectors_next_iter = []
            shape = [batch_size, -1, self.sample_num, self.dim]
            for hop in range(self.hop - n_hop):
                aggregator = self.global_agg[n_hop]
                vector = aggregator(self_vectors=entity_vectors_[hop],
                                    neighbor_vector=[entity_vectors_[hop+1][k].view(shape) for k in range(len(entity_vectors_[hop + 1]))],
                                    masks=None,
                                    batch_size=batch_size,
                                    neighbor_weight=[weight_vectors[hop][k].view(batch_size, -1, self.sample_num) for k in range(len(weight_vectors[hop]))],
                                    pos_weight=[pos_vectors[hop][k].view(batch_size, -1, self.sample_num, self.hiddenSize_pos) for k in range(len(pos_vectors[hop]))],
                                    extra_vector=session_info[hop])
                entity_vectors_next_iter.append(vector)
            entity_vectors_ = entity_vectors_next_iter
        for i in range(len(entity_vectors_)):
            h_global = entity_vectors_[i][0].view(batch_size, seqs_len, self.dim)



        # combine


        output = [h_local,h_global]

        return output


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward(model, data):
    alias_inputs, adj, items, mask, targets, inputs = data
    alias_inputs = trans_to_cuda(alias_inputs).long()
    items = trans_to_cuda(items).long()
    adj = trans_to_cuda(adj).float()
    mask = trans_to_cuda(mask).long()
    inputs = trans_to_cuda(inputs).long()

    hidden = model(items, adj, mask, inputs)
    # get = lambda index: hidden[index][alias_inputs[index]]
    # seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
    get_local = lambda index: hidden[0][index][alias_inputs[index]]
    seq_hidden_local = torch.stack([get_local(i) for i in torch.arange(len(alias_inputs)).long()])
    get_global = lambda index: hidden[1][index][alias_inputs[index]]
    seq_hidden_global = torch.stack([get_global(i) for i in torch.arange(len(alias_inputs)).long()])
    scores,conloss=model.compute_scores(seq_hidden_local,seq_hidden_global,mask)
    # return targets, model.compute_scores(seq_hidden, mask)
    return targets,scores,conloss


def train_test(model, train_data, test_data):
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    total_conloss = 0.0
    train_loader = torch.utils.data.DataLoader(train_data, num_workers=4, batch_size=model.batch_size,
                                               shuffle=True, pin_memory=True)
    for data in tqdm(train_loader):
        model.optimizer.zero_grad()
        targets, scores, conloss = forward(model, data)
        targets = trans_to_cuda(targets).long()
        loss = model.loss_function(scores, targets - 1)
        loss = loss + conloss
        loss.backward()
        model.optimizer.step()
        total_loss += loss
        total_conloss += conloss
    print('\tLoss:\t%.3f' % total_loss)
    print('\tconloss:\t%.3f' % total_conloss)
    model.scheduler.step()

    print('start predicting: ', datetime.datetime.now())
    model.eval()
    test_loader = torch.utils.data.DataLoader(test_data, num_workers=4, batch_size=model.batch_size,
                                              shuffle=False, pin_memory=True)
    result = []
    hit, mrr = [], []
    for data in test_loader:
        targets, scores,_ = forward(model, data)
        sub_scores = scores.topk(20)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        targets = targets.numpy()
        for score, target, mask in zip(sub_scores, targets, test_data.mask):
            hit.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))

    result.append(np.mean(hit) * 100)
    result.append(np.mean(mrr) * 100)

    return result
