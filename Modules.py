import copy
import dgl
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):

    def __init__(self, heads, emb_size, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.heads = heads
        self.emb_size = emb_size
        self.q_linear = nn.Linear(emb_size, emb_size)
        self.k_linear = nn.Linear(emb_size, emb_size)
        self.v_linear = nn.Linear(emb_size, emb_size)
        self.dropout = nn.Dropout(dropout)
        self.apply_dropout = True

        self.out = nn.Linear(emb_size, emb_size)

    def forward(self, q, k, v):
        k = self.k_linear(k)
        q = self.q_linear(q)
        v = self.v_linear(v)
        scores = self.attention(q, k, v)
        concat = scores
        output = self.out(concat)
        return output

    def attention(self, q, k, v):
        scores = torch.matmul(q, k.transpose(-2, -1 )) / math.sqrt(self.emb_size)
        scores = torch.softmax(scores, dim=-1)
        if self.apply_dropout:
            scores = self.dropout(scores)
        output = torch.matmul(scores, v)
        return output


class FeedForward(nn.Module):
    def __init__(self, emb_size, h_size=512, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(emb_size, h_size)
        self.linear2 = nn.Linear(h_size, emb_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class Norm(nn.Module):
    def __init__(self, emb_size, eps=1e-6):
        super(Norm, self).__init__()
        self.emb_size = emb_size
        self.alpha = nn.Parameter(torch.ones(self.emb_size))
        self.bias = nn.Parameter(torch.zeros(self.emb_size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


class Embedder(nn.Module):

    def __init__(self, num_nodes, emb_size):
        super(Embedder, self).__init__()
        self.embed = nn.Embedding(num_nodes, emb_size)

    def forward(self, x):
        return self.embed(x)


class RelGraphEmbed(nn.Module):
    r"""Embedding layer for featureless heterograph."""
    def __init__(self, g, embed_size, embed_name='embed', pretrained=None):
        super(RelGraphEmbed, self).__init__()
        self.g = g
        self.embed_size = embed_size
        self.embed_name = embed_name

        # create weight embeddings for each node for each relation
        # self.embeds = nn.ParameterDict()
        self.embeds = {}
        for ntype in g.ntypes:
            if pretrained is None:
                embed = nn.Parameter(torch.Tensor(g.number_of_nodes(ntype), self.embed_size))
                nn.init.xavier_uniform_(embed, gain=nn.init.calculate_gain('relu'))
                self.embeds[ntype] = embed
            elif pretrained == ntype:
                embed = g.nodes[pretrained].data['feat']
                self.embeds[ntype] = embed

    def forward(self, input_nodes, eps=1e-12):
        emb = {}
        for ntype, nid in input_nodes.items():
            nid = input_nodes[ntype]
            res = self.embeds[ntype][nid].clone()
            # res[node_embed[ntype][nid] == 0] = eps
            emb[ntype] = res
        return emb


def get_clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


"""
    MLP Layer used after graph vector representation
"""
class PreReadout(nn.Module):

    def __init__(self, input_dim):
        super(PreReadout, self).__init__()
        self.att = nn.TransformerEncoderLayer(d_model=input_dim, nhead=8, batch_first=True)

    def forward(self, g):
        gs = dgl.unbatch(g)
        max_length = max([bg.ndata['res'].shape[0] for bg in gs])
        tmp = [F.pad(bg.ndata['res'], pad=(0, 0, 0, max_length - bg.ndata['res'].shape[0]), value=0) for bg in gs]
        x = torch.stack(tmp)

        x = self.att(x)
        x = torch.vstack([entry[:gs[idx].ndata['res'].size()[0],:] for idx, entry in enumerate(x)])
        g.ndata['res'] = x
        return x


class MLPReadout(nn.Module):

    def __init__(self, input_dim, output_dim, L=2):  # L=nb_hidden_layers
        super().__init__()
        list_FC_layers = [nn.Linear(input_dim // 2 ** l, input_dim // 2 ** (l + 1), bias=False) for l in range(L)]
        list_FC_layers.append(nn.Linear(input_dim // 2 ** L, output_dim, bias=False))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.prelu = nn.PReLU()
        self.L = L

    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = self.prelu(y)
        y = self.FC_layers[self.L](y)
        return y

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            m.bias.data.fill_(0.01)
