from utils import EpsilonGreedyStrategy
import torch
import torch.nn as nn
import TBase
import random
from Modules import get_clones
import numpy as np
import dgl
import torch.functional as F
import math


class Encoder(TBase.Encoder):

    def __init__(self, params, n_entities=None, n_rels=None):
        super(Encoder, self).__init__(params, n_entities, n_rels)
        self.params = params
        self.num_experts = params['num_experts']
        self.routing_h = nn.Linear(self.emb_size, self.num_experts)
        self.experts_h = get_clones(nn.Linear(self.emb_size, self.emb_size), self.num_experts)
        self.apply_scaling = params['apply_scaling']

        # Greedy expert exploration
        self.apply_greedy = params['apply_greedy']
        self.strategy = EpsilonGreedyStrategy(1.0, 0.05, 5e-2)
        self.current_step = 0

        # self.attention_v_linear = get_clones(nn.Linear(self.emb_size, self.emb_size), self.num_experts)
        # self.attention_q_linear = get_clones(nn.Linear(self.emb_size, self.emb_size), self.num_experts)
        # self.attention_k_linear = get_clones(nn.Linear(self.emb_size, self.emb_size), self.num_experts)
        # self.attention_v_linear = nn.Linear(self.emb_size, self.emb_size)
        self.attention_q_linear = nn.Linear(self.emb_size, self.emb_size, bias=False)
        self.attention_k_linear = nn.Linear(self.emb_size, self.emb_size, bias=False)

        torch.nn.init.xavier_uniform_(self.attention_q_linear.weight)
        torch.nn.init.xavier_uniform_(self.attention_k_linear.weight)

    def forward(self, g, h, e, useOGBMolEncoder=True, graph_readout=True, h_lap_pos_enc=None):
        if useOGBMolEncoder:
            h = self.embedding_n(h)
            e = self.embedding_e(e)
        else:
            h = self.emb_n(h.int())
            e = self.emb_e(e.int())
            g.ndata['it_feat'] = h

        h_in = h
        if self.lap_pos_enc:
            h_lap_pos_enc = self.embedding_lap_pos_enc(h_lap_pos_enc.float())
            h = h + h_lap_pos_enc

        # STANDARD SOLUTION
        # ROUTING MECHANISM - SINGLE EXPERT
        routing_h = torch.softmax(self.routing_h(h), dim=-1)
        rate = self.strategy.get_exploration_rate(self.current_step)
        if self.apply_greedy and rate > random.random():
            self.current_step += 1
            rout_h_max_idx = np.random.choice(np.arange(0, self.num_experts), h.size(0), 1)
            rout_h_max_idx = torch.tensor(rout_h_max_idx, device=h.device)
            rout_h_max_vals = torch.gather(routing_h, 1, rout_h_max_idx.unsqueeze(-1)).squeeze().to(h.device)
        else:
            rout_h_max_vals, rout_h_max_idx = torch.max(routing_h, dim=1)
        assert self.num_experts == len(self.layers), "Number of experts should be equal to the number of shells"

        for layer_idx, layer in enumerate(self.layers):
            h, e = self.layers[layer_idx](g, h, e, h_in)
            g.ndata['res_{}'.format(layer_idx)] = h

        h_tmp = torch.full_like(h, 0)

        for unique_value in torch.unique(rout_h_max_idx):
            sliced_h = self.experts_h[unique_value](g.ndata["res_{}".format(unique_value)].view(-1, self.emb_size)[rout_h_max_idx == unique_value])
            if self.apply_scaling:
                sliced_h = sliced_h * rout_h_max_vals[rout_h_max_idx == unique_value].unsqueeze(-1)
            h_tmp[rout_h_max_idx == unique_value] = sliced_h # residual : + h_in[rout_h_max_idx == unique_value]

        g.ndata['res'] = h_tmp

        if graph_readout:
            hg = self.readout_fnc(g, 'res')
            out = self.out_layer(hg)
        else:
            out = self.out_layer(h)
        return out, rout_h_max_idx
