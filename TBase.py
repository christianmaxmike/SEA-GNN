import dgl
import torch.nn
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from Modules import get_clones, MLPReadout, PreReadout
import dgl.function as fn
from utils import *
import numpy as np


class MultiHeadAttention(nn.Module):

    def __init__(self, in_channels, out_channels, heads,
                 bias=False, full_graph=False, twohop=False, aug=False, agg_fnc="sum"):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = heads
        self.in_dim = in_channels
        self.out_dim = out_channels
        self.full_graph = full_graph
        self.twohop = twohop
        self.aug = aug
        self.gamma = 1e-5

        self._in_src_feats = self.out_dim * self.num_heads

        self.q_linear = nn.Linear(self.in_dim, self.out_dim * self.num_heads, bias=bias)
        self.k_linear = nn.Linear(self.in_dim, self.out_dim * self.num_heads, bias=bias)
        self.v_linear = nn.Linear(self.in_dim, self.out_dim * self.num_heads, bias=bias)
        self.e_linear = nn.Linear(self.in_dim, self.out_dim * self.num_heads, bias=bias)
        if self.full_graph or (self.twohop and self.aug):
            self.Q_2 = nn.Linear(self.in_dim, self.out_dim * self.num_heads, bias=bias)
            self.K_2 = nn.Linear(self.in_dim, self.out_dim * self.num_heads, bias=bias)
            self.E_2 = nn.Linear(self.in_dim, self.out_dim * self.num_heads, bias=bias)
            self.V_2 = nn.Linear(self.in_dim, self.out_dim * self.num_heads, bias=bias)

        if agg_fnc == 'sum':
            self.agg_fnc = fn.sum
        elif agg_fnc == 'mean':
            self.agg_fnc = fn.mean
        elif agg_fnc == 'max':
            self.agg_fnc = fn.max
        elif agg_fnc == 'lstm':
            self.agg_fnc = self.lstm_reducer
            self.lstm = nn.LSTM(self._in_src_feats, self._in_src_feats, batch_first=True)
        else:
            raise NotImplementedError("Unknown aggregation function: {}".format(self.agg_fnc))


    def forward(self, g, h, e, h_in, alpha_val=None):
        # Linear Modules
        k = self.k_linear(h)
        e = self.e_linear(e)
        q = self.q_linear(h)
        v = self.v_linear(h_in)

        if self.full_graph or (self.twohop and self.aug):
            Q_2h = self.Q_2(h)
            K_2h = self.K_2(h)
            E_2 = self.E_2(e)
            V_2h = self.V_2(h_in)

        # Scaled Dot-Product Attention
        g.ndata['V_h'] = v.view(-1, self.num_heads, self.out_dim)
        g.ndata['Q_h'] = q.view(-1, self.num_heads, self.out_dim)
        g.ndata['K_h'] = k.view(-1, self.num_heads, self.out_dim)
        g.edata['E'] = e.view(-1, self.num_heads, self.out_dim)

        if self.full_graph or (self.twohop and self.aug):
            g.ndata['Q_2h'] = Q_2h.view(-1, self.num_heads, self.out_dim)
            g.ndata['K_2h'] = K_2h.view(-1, self.num_heads, self.out_dim)
            # g.edata['E_2'] = E_2.view(-1, self.num_heads, self.out_dim)
            # g.ndata['V_2h'] = V_2h.view(-1, self.num_heads, self.out_dim)

        self.graph_dynamics(g, alpha_val)
        # self.graph_dynamics_pos(g)
        # self.graph_dynamics_neg(g)

        h_out = g.ndata['aV'] / (g.ndata['z'] + torch.full_like(g.ndata['z'], 1e-6))
        # h_out_p = g.ndata['aVp'] / (g.ndata['zp'] + torch.full_like(g.ndata['zp'], 1e-6))
        # h_out_n = g.ndata['aVn'] / (g.ndata['zn'] + torch.full_like(g.ndata['zn'], 1e-6))
        # h_out = torch.cat((h_out_p,h_out_n), dim=-1)

        e_out = g.edata['E'].view(-1, self.num_heads * self.out_dim)
        return h_out, e_out

    def lstm_reducer(self, in_field, out_field):
        def _lstm_reducer(nodes):
            """LSTM reducer
            lstm reducer with default schedule (degree bucketing)
            """
            m = nodes.mailbox[in_field]  # .squeeze()  # (B, L, D)
            m = m.reshape(m.shape[0], m.shape[1], -1)
            m = m[:, torch.randperm(m.size()[1]), :]
            batch_size = m.shape[0]
            h = (m.new_zeros((1, batch_size, self._in_src_feats)),
                 m.new_zeros((1, batch_size, self._in_src_feats)))
            _, (rst, _) = self.lstm(m, h)
            rst = rst.squeeze(0)
            rst = rst.reshape(-1, self.num_heads, self.out_dim)
            return {out_field: rst}
        return _lstm_reducer

    def graph_dynamics_pos(self, g):
        real_ids = torch.nonzero(g.edata['real']).squeeze()
        g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score'), edges=real_ids)
        g.apply_edges(scaling('score', math.sqrt(self.out_dim)), edges=real_ids)
        g.apply_edges(imp_exp_attn('score', 'E'), edges=real_ids)
        g.apply_edges(exp_real('score', 'score_soft', self.gamma), edges=real_ids)
        g.send_and_recv(real_ids, fn.src_mul_edge('V_h', 'score_soft', 'V_h'), self.agg_fnc('V_h', 'aVp'))
        g.send_and_recv(real_ids, fn.copy_edge('score_soft', 'score_soft'), fn.sum('score_soft', 'zp'))

    def graph_dynamics_neg(self, g):
        fake_ids = torch.nonzero(g.edata['real'] == 0).squeeze()
        g.apply_edges(src_dot_dst('K_2h', 'Q_2h', 'score'), edges=fake_ids)
        g.apply_edges(scaling('score', np.sqrt(self.out_dim)))
        g.apply_edges(imp_exp_attn('score', 'E_2'), edges=fake_ids)
        g.apply_edges(exp_fake('score', 'score_soft', self.gamma), edges=fake_ids)
        g.send_and_recv(fake_ids, fn.src_mul_edge('V_2h', 'score_soft', 'V_2h'), self.agg_fnc('V_2h', 'aVn'))
        g.send_and_recv(fake_ids, fn.copy_edge('score_soft', 'score_soft'), fn.sum('score_soft', 'zn'))

    def graph_dynamics(self, g, alpha_val=None):

        if self.full_graph or (self.twohop and self.aug):
            real_ids = torch.nonzero(g.edata['real']).squeeze()
            fake_ids = torch.nonzero(g.edata['real'] == 0).squeeze()
        else:
            real_ids = g.edges(form='eid')

        if alpha_val is not None:
            gs = dgl.unbatch(g)
            scores = []
            for b_g in gs:
                b_g_edges = b_g.edges(form='eid')
                b_g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score', alpha_val), edges=b_g_edges)
                scores.append(b_g.edata['score'])
            # g = dgl.batch(gs)
            g.edata['score'] = torch.vstack(scores)
        else:
            g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score'), edges=real_ids)

        if self.full_graph or (self.twohop and self.aug):
            g.apply_edges(src_dot_dst('K_2h', 'Q_2h', 'score'), edges=fake_ids)

        # scale scores by sqrt(d)
        g.apply_edges(scaling('score', np.sqrt(self.out_dim)))

        # Use available edge features to modify the scores for edges
        g.apply_edges(imp_exp_attn('score', 'E'), edges=real_ids)

        if self.full_graph or (self.twohop and self.aug):
            g.apply_edges(imp_exp_attn('score', 'E_2'), edges=fake_ids)

        if self.full_graph or (self.twohop and self.aug):
            # softmax and scaling by gamma
            L = self.gamma
            g.apply_edges(exp_real('score', 'score_soft', L), edges=real_ids)
            g.apply_edges(exp_fake('score', 'score_soft', L), edges=fake_ids)

        else:
            g.apply_edges(exp('score'), edges=real_ids)

        # Send weighted values to target nodes
        eids = g.edges()

        g.send_and_recv(eids, fn.src_mul_edge('V_h', 'score_soft', 'V_h'), self.agg_fnc('V_h', 'aV'))
        g.send_and_recv(eids, fn.copy_edge('score_soft', 'score_soft'), fn.sum('score_soft', 'z'))

        # h_out = g.ndata['aV'] / (g.ndata['z'] + torch.full_like(g.ndata['z'], 1e-6))
        # e_out = g.edata['E'].view(-1, self.num_heads * self.out_dim)

        # return h_out, e_out

    def graph_dynamics_old(self, g):
        edge_ids = g.edges(form='eid')
        # MatMul
        g.apply_edges(src_dot_dst('K', 'Q', 'score'), edges=edge_ids)
        # Scale
        g.apply_edges(scaling('score', math.sqrt(self.out_dim)), edges=edge_ids)
        # incorporate additional edge information
        g.apply_edges(imp_exp_attn('score', 'E'), edges=edge_ids)
        # exp; stores in 'score_soft'
        g.apply_edges(exp('score'), edges=edge_ids)
        # compute nominator of softmax times V:values
        g.send_and_recv(edge_ids, fn.src_mul_edge('V', 'score_soft', 'V'), self.agg_fnc('V', 'aV'))
        # Compute denominator for softmax
        g.send_and_recv(edge_ids, fn.copy_edge('score_soft', 'score_soft'), fn.sum('score_soft', 'z'))


class EncoderLayer(nn.Module):

    def __init__(self, in_dim, out_dim, num_heads, dropout,
                 layer_norm=False, batch_norm=True, residual=True, edge_learning=False,
                 use_bias=False, full_graph=False, twohop=False, aug=False, agg_fnc='sum'):
        super(EncoderLayer, self).__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.residual = residual
        self.edge_learning = edge_learning

        self.attention = MultiHeadAttention(self.in_channels, self.out_channels//self.num_heads, self.num_heads,
                                            bias=use_bias, full_graph=full_graph, twohop=twohop, aug=aug, agg_fnc=agg_fnc)
        self.sdp_layer = nn.Linear(self.out_channels, self.out_channels)
        self.out_h = nn.Linear(out_dim, out_dim * 2)
        self.out_h_2 = nn.Linear(out_dim * 2, out_dim)

        self.dropout = nn.Dropout(dropout)
        if self.edge_learning:
            self.out_e = nn.Linear(out_dim, out_dim)
        if self.layer_norm:
            self.layer_norm_1 = nn.LayerNorm(out_dim)
        if self.batch_norm:
            self.batch_norm_1 = nn.BatchNorm1d(out_dim)


    def forward(self, g, h, e, h_in, alpha_val=None, layer_id=None):
        h_in = h
        e_in = e

        # Self-attention
        h_attn, e_out = self.attention(g, h, e, h_in, alpha_val)
        # Concat
        h_out = h_attn.view(-1, self.out_channels)
        # h_out = h_attn.view(-1, self.out_channels * 2)
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


class Encoder(nn.Module):
    def __init__(self, params, n_entities=None, n_rels=None):
        super(Encoder, self).__init__()
        self.num_heads = params['num_heads']
        self.emb_size = params['emb_dim']
        self.dropout = params['dropout']
        self.n = params['N']
        self.n_classes = params['no_classes']
        self.num_experts = 1  # default GTL


        self.embedding_n = AtomEncoder(emb_dim=self.emb_size)
        self.embedding_e = BondEncoder(emb_dim=self.emb_size)
        if n_entities is not None and n_rels is not None:
            self.emb_n = nn.Embedding(n_entities, self.emb_size)
            self.emb_e = nn.Embedding(n_rels, self.emb_size)

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
                                              params['aug'],
                                              params['agg_fnc']), self.n)

        self.lap_pos_enc = params['lap_pos_enc']
        if self.lap_pos_enc:
            pos_enc_dim = params['pos_enc_dim']
            self.embedding_lap_pos_enc = nn.Linear(pos_enc_dim, self.emb_size)

        self.readout = params['readout_fnc']
        if self.readout == "sum":
            self.readout_fnc = dgl.sum_nodes
        elif self.readout == "mean":
            self.readout_fnc = dgl.mean_nodes
        elif self.readout == "max":
            self.readout_fnc = dgl.max_nodes
        else:
            raise NotImplementedError("Unknown readout function: {}".format(self.readout))
        # self.out_layer = MLPReadout(self.emb_size, params['no_classes'])
        self.out_layer = nn.Linear(self.emb_size, params['no_classes'])
        self.out_act_fnc = nn.Sigmoid()
        # self.prereadout = PreReadout(self.emb_size)

    def forward(self, g, h, e, useOGBMolEncoder=True, graph_readout=True, h_lap_pos_enc=None):
        if useOGBMolEncoder:
            h = self.embedding_n(h)
            e = self.embedding_e(e)
        else:
            h = self.emb_n(h.int())
            e = self.emb_e(e.int())
        # h = self.in_feat_dropout(h)
        if self.lap_pos_enc:
            h_lap_pos_enc = self.embedding_lap_pos_enc(h_lap_pos_enc.float())
            h = h + h_lap_pos_enc

        for idx, layer in enumerate(self.layers):
            h, e = self.layers[idx](g, h, e)
        g.ndata['res'] = h

        if graph_readout:
            hg = self.readout_fnc(g, 'res')
            out = self.out_layer(hg)
        else:
            out = self.out_layer(h)
        return out, None

    def loss_fnc(self, type, scores, targets):
        if type == "BCELoss":
            scores = self.out_act_fnc(scores)
            loss = torch.nn.BCELoss()(scores.float(), targets.float())
        elif type == "BCEWithLogitsLoss":
            loss = torch.nn.BCEWithLogitsLoss()(scores.float(), targets.float())
        elif type == "L1Loss":
            loss = torch.nn.L1Loss()(scores.float(), targets.float())
        elif type == "MSELoss":
            loss = torch.nn.MSELoss()(scores.float(), targets.float())
        elif type == "SBM_CrossEntropy":
            # calculating label weights for weighted loss computation
            V = targets.size(0)
            label_count = torch.bincount(targets)
            label_count = label_count[label_count.nonzero()].squeeze()
            cluster_sizes = torch.zeros(self.n_classes).long().to(scores.device)
            cluster_sizes[torch.unique(targets)] = label_count
            weight = (V - cluster_sizes).float() / V
            weight *= (cluster_sizes > 0).float()
            # weighted cross-entropy for unbalanced classes
            criterion = nn.CrossEntropyLoss(weight=weight)
            loss = criterion(scores, targets)
        else:
            raise AssertionError("Unknown loss function")
        return loss
