import utils
import dgl.function as fn
from utils import *
import TBase
import random
from Modules import get_clones

import numpy as np


class MultiHeadAttention(TBase.MultiHeadAttention):

    def __init__(self, in_channels, out_channels, heads, bias=False, full_graph=False, twohop=False, aug=False):
        super(MultiHeadAttention, self).__init__(in_channels, out_channels, heads, bias, full_graph, twohop, aug)

    def forward(self, g, h, e, layer_idx=None):
        # Linear Modules
        k = self.k_linear(h)
        e = self.e_linear(e)
        q = self.q_linear(h)
        v = self.v_linear(h)

        if self.full_graph or (self.twohop and self.aug):
            Q_2h = self.Q_2(h)
            K_2h = self.K_2(h)
            E_2 = self.E_2(e)
            V_2h = self.V_2(h)

        # Scaled Dot-Product Attention
        g.ndata['V_h'] = v.view(-1, self.num_heads, self.out_dim)
        #if layer_idx==0:
        g.ndata['Q_h'] = q.view(-1, self.num_heads, self.out_dim)
        g.ndata['K_h'] = k.view(-1, self.num_heads, self.out_dim)
        g.edata['E'] = e.view(-1, self.num_heads, self.out_dim)

        if self.full_graph or (self.twohop and self.aug):
            #if layer_idx == 0:
            g.ndata['Q_2h'] = Q_2h.view(-1, self.num_heads, self.out_dim)
            g.ndata['K_2h'] = K_2h.view(-1, self.num_heads, self.out_dim)
            g.edata['E_2'] = E_2.view(-1, self.num_heads, self.out_dim)
            #g.ndata['V_2h'] = V_2h.view(-1, self.num_heads, self.out_dim)

        self.graph_dynamics(g)
        # self.graph_dynamics_pos(g)
        # self.graph_dynamics_neg(g)

        h_out = g.ndata['aV'] / (g.ndata['z'] + torch.full_like(g.ndata['z'], 1e-6))
        # h_out_p = g.ndata['aVp'] / (g.ndata['zp'] + torch.full_like(g.ndata['zp'], 1e-6))
        # h_out_n = g.ndata['aVn'] / (g.ndata['zn'] + torch.full_like(g.ndata['zn'], 1e-6))
        # h_out = torch.cat((h_out_p,h_out_n), dim=-1)
        e_out = g.edata['E'].view(-1, self.num_heads * self.out_dim)
        return h_out, e_out


class EncoderLayer(TBase.EncoderLayer):

    def __init__(self, in_dim, out_dim, num_heads, dropout,
                 layer_norm=True, batch_norm=True, residual=True, edge_learning=False,
                 use_bias=False, full_graph=False, twohop=False, aug=False):
        super(EncoderLayer, self).__init__(in_dim, out_dim, num_heads, dropout,
                                           layer_norm, batch_norm, residual, edge_learning,
                                           use_bias, full_graph, twohop, aug)
        self.attention = MultiHeadAttention(self.in_channels, self.out_channels//self.num_heads, self.num_heads,
                                            bias=use_bias, full_graph=full_graph, twohop=twohop, aug=aug)


    def forward(self, g, h, e, layer_id=None):
        h_in = h
        e_in = e

        # Self-attention
        h_attn, e_out = self.attention(g, h, e, layer_id)
        # Concat
        h_out = h_attn.view(-1, self.out_channels)
        # Dropout
        h_out = self.dropout(h_out)
        # Linear
        h = self.sdp_layer(h_out)

        if self.residual:
            h = h_in + h
        if self.layer_norm:
            h = self.layer_norm_1(h)
        if self.batch_norm:
            h = self.batch_norm_1(h)

        # linear layer
        h_in2 = h

        h = self.out_h(h)
        h = torch.relu(h)
        h = self.dropout(h)
        h = self.out_h_2(h)

        if self.residual:
            h = h_in2 + h
        if self.layer_norm:
            h = self.layer_norm_1(h)
        if self.batch_norm:
            h = self.batch_norm_1(h)

        if self.edge_learning:
            if self.residual:
                e = e_in + e_out
            if self.layer_norm:
                e = self.layer_norm_1(e)
            if self.batch_norm:
                e = self.batch_norm_1(e)
            e_in2 = e
            e = self.out_e(e)
            if self.residual:
                e = e + e_in2
            if self.layer_norm:
                e = self.layer_norm_1(e)
            if self.batch_norm:
                e = self.batch_norm_1(e)
        return h, e


class Encoder(TBase.Encoder):

    def __init__(self, params, n_entities=None, n_rels=None):
        super(Encoder, self).__init__(params, n_entities, n_rels)
        self.params = params
        self.num_experts = params['num_experts']
        self.layers = get_clones(EncoderLayer(self.emb_size,
                                              self.emb_size,
                                              self.num_heads,
                                              self.dropout,
                                              params['layer_norm'],
                                              params['batch_norm'],
                                              params['residual'],
                                              params['edge_learning'],
                                              params['use_bias'],
                                              params['full_graph'],
                                              params['k_hop'],
                                              params['aug']), self.n)

        self.routing_h = nn.Linear(self.emb_size, self.num_experts)
        self.experts_h = get_clones(nn.Linear(self.emb_size, self.emb_size), self.num_experts)
        self.apply_scaling = params['apply_scaling']

        # Greedy expert exploration
        self.apply_greedy = params['apply_greedy']
        self.strategy = utils.EpsilonGreedyStrategy(1.0, 0.05, 5e-2)
        self.current_step = 0

    def forward(self, g, h, e, useOGBMolEncoder=True, graph_readout=True, h_lap_pos_enc=None):
        if useOGBMolEncoder:
            h = self.embedding_n(h)
            e = self.embedding_e(e)
        else:
            h = self.emb_n(h.int())
            e = self.emb_e(e.int())

        if self.lap_pos_enc:
            h_lap_pos_enc = self.embedding_lap_pos_enc(h_lap_pos_enc.float())
            h = h + h_lap_pos_enc

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

        # node_ids = g.nodes()
        # if self.params['full_graph']:
        #     real_ids = torch.nonzero(g.edata['real']).squeeze()
        #     fake_ids = torch.nonzero(g.edata['real'] == 0).squeeze()
        # else:
        #     real_ids = g.edges(form='eid')

        eids = g.edges()
        for layer_idx, layer in enumerate(self.layers):
            h, e = self.layers[layer_idx](g, h, e, layer_idx)
            g.ndata['res_{}'.format(layer_idx)] = h
            # pull-push-split
            # g.push(node_ids, fn.copy_u("aV", "aV"), fn.sum('aV', 'Q_h'))
            # g.push(node_ids, fn.copy_u("aV", "aV"), fn.sum('aV', 'Q_2h'))
            g.send_and_recv(eids, fn.copy_u("res_{}".format(layer_idx), "res_{}".format(layer_idx)),
                            fn.mean("res_{}".format(layer_idx), 'htmp'))
            h = g.ndata['htmp']

        h_tmp = torch.full_like(h, 0)

        for unique_value in torch.unique(rout_h_max_idx):
            sliced_h = self.experts_h[unique_value](g.ndata["res_{}".format(unique_value)].view(-1, self.emb_size)[rout_h_max_idx == unique_value])
            if self.apply_scaling:
                sliced_h = sliced_h * rout_h_max_vals[rout_h_max_idx == unique_value].unsqueeze(-1)
            h_tmp[rout_h_max_idx == unique_value] = sliced_h
        g.ndata['res'] = h_tmp

        if graph_readout:
            hg = self.readout_fnc(g, 'res')
            out = self.out_layer(hg)
        else:
            out = self.out_layer(h)
        return out, rout_h_max_idx
