from abc import ABC

import dgl
import torch.utils.data
from ogb.graphproppred import DglGraphPropPredDataset
import os
import time
import pickle
import csv
from tqdm import tqdm
from dgl.data.utils import save_graphs, load_graphs
import networkx as nx
import numpy as np
from scipy import sparse as sp
import torch.nn.functional as F
from sys import float_info


class GraphPropLoader(torch.utils.data.Dataset):

    def __init__(self, name):
        start = time.time()
        print("[I] Loading dataset %s..." % name)
        self.name = name

        dataset = DglGraphPropPredDataset(name=name)
        split_idx = dataset.get_idx_split()

        split_idx["train"] = split_idx["train"]
        split_idx["valid"] = split_idx["valid"]
        split_idx["test"] = split_idx["test"]

        self.train = dataset[split_idx["train"]]
        self.valid = dataset[split_idx["valid"]]
        self.test = dataset[split_idx["test"]]

        print('train, test, valid sizes :', len(self.train), len(self.test), len(self.valid))
        print("[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time() - start))

    def collate_dgl(self, samples):
        """ Code as explained in API dgl"""
        graphs, labels = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        labels = torch.stack(labels)
        return batched_graph, labels


class MolPCBA(GraphPropLoader, ABC):

    def __init__(self, name="ogbg-molpcba"):
        super(MolPCBA, self).__init__(name)


class MolHIVDataset(torch.utils.data.Dataset):
    def __init__(self, name='ogbg-molhiv'):
        start = time.time()
        print("[I] Loading dataset %s..." % name)
        self.name = name

        dataset = DglGraphPropPredDataset(name=name)
        split_idx = dataset.get_idx_split()

        split_idx["train"] = split_idx["train"]
        split_idx["valid"] = split_idx["valid"]
        split_idx["test"] = split_idx["test"]

        self.train = dataset[split_idx["train"]]
        self.valid = dataset[split_idx["valid"]]
        self.test = dataset[split_idx["test"]]

        print('train, test, valid sizes :', len(self.train), len(self.test), len(self.valid))
        print("[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time() - start))

    def collate_dgl(self, samples):
        """ Code as explained in dgl's API """
        graphs, labels = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        labels = torch.stack(labels)
        return batched_graph, labels

    def _add_laplacian_positional_encodings(self, pos_enc_dim):
        # Graph positional encoding w/ Laplacian eigenvectors
        print("Laplacian Positional Encoding: Start pre-processing graphs...")
        print("train set ...")
        self.train.dataset.graphs = [self._laplacian_positional_encoding(g, pos_enc_dim)
                                     for g in self.train.dataset.graphs]
        print("valid set ...")
        self.valid.dataset.graphs = [self._laplacian_positional_encoding(g, pos_enc_dim)
                                     for g in self.valid.dataset.graphs]
        print("test set ...")
        self.test.dataset.graphs = [self._laplacian_positional_encoding(g, pos_enc_dim)
                                    for g in self.test.dataset.graphs]
        print("LPE processing finished")

    def _laplacian_positional_encoding(self, g, pos_enc_dim):
        """
            Graph positional encoding v/ Laplacian eigenvectors
        """
        # Laplacian
        n = g.number_of_nodes()
        A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
        N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
        L = sp.eye(g.number_of_nodes()) - N * A * N

        # Eigenvectors with numpy
        eigenvalues, eigenvectors = np.linalg.eig(L.toarray())
        idx = eigenvalues.argsort()  # increasing order
        eigenvalues, eigenvectors = eigenvalues[idx],  np.real(eigenvectors[:, idx])

        eigenvectors = torch.from_numpy(eigenvectors).float()
        eigenvectors = F.normalize(eigenvectors, p=2, dim=1, eps=1e-12, out=None)

        if n < pos_enc_dim:
            eigenvectors = F.pad(eigenvectors, (0, pos_enc_dim - n), value=float_info.epsilon)  # value=float('nan'))

        g.ndata['lap_pos_enc'] = eigenvectors[:, :pos_enc_dim]

        return g

    def _addkhops(self, g, k_cutoff=2):
        paths = dict(nx.all_pairs_shortest_path(nx.Graph(dgl.to_networkx(g)), cutoff=k_cutoff))
        # remove self-loops
        paths = {key: {inner_key: path for inner_key, path in v.items() if inner_key != key} for key, v in paths.items()}
        twohop_g = nx.Graph(paths)
        full_g = dgl.from_networkx(twohop_g)

        # Here we copy over the node feature data and laplace encodings
        full_g.ndata['feat'] = g.ndata['feat']

        try:
            full_g.ndata['EigVecs'] = g.ndata['EigVecs']
            full_g.ndata['EigVals'] = g.ndata['EigVals']
        except:
            pass

        # Populate edge features w/ 0s
        full_g.edata['feat'] = torch.zeros(full_g.number_of_edges(), 3, dtype=torch.long)
        full_g.edata['real'] = torch.zeros(full_g.number_of_edges(), dtype=torch.long)

        # Copy real edge data over
        full_g.edges[g.edges(form='uv')[0].tolist(), g.edges(form='uv')[1].tolist()].data['feat'] = g.edata['feat']
        full_g.edges[g.edges(form='uv')[0].tolist(), g.edges(form='uv')[1].tolist()].data['real'] = torch.ones(
            g.edata['feat'].shape[0], dtype=torch.long)

        return full_g

    def addkhops(self, k_cutoff=2):
        print("Add {} hops: start pre-processing graphs ...".format(k_cutoff))
        print("train set ...")
        self.train.dataset.graphs = [self._addkhops(g, k_cutoff) for g in tqdm(self.train.dataset.graphs)]
        print("valid set ...")
        self.valid.dataset.graphs = [self._addkhops(g, k_cutoff) for g in tqdm(self.valid.dataset.graphs)]
        print("test set ...")
        self.test.dataset.graphs = [self._addkhops(g, k_cutoff) for g in tqdm(self.test.dataset.graphs)]
        print("{}-hops successfully added".format(k_cutoff))


class MoleculeDGL(torch.utils.data.Dataset):
    def __init__(self, data_dir, split, num_graphs=None):
        """
        data is a list of Molecule dict objects with following attributes

          molecule = data[idx]
        ; molecule['num_atom'] : nb of atoms, an integer (N)
        ; molecule['atom_type'] : tensor of size N, each element is an atom type, an int between 0 and num_atom_type
        ; molecule['bond_type'] : tensor of size N x N, each element is a bond type, an int between 0 and num_bond_type
        ; molecule['logP_SA_cycle_normalized'] : the chemical property to regress, a float variable
        """

        self.data_dir = data_dir
        self.split = split
        self.num_graphs = num_graphs

        with open(data_dir + "/%s.pickle" % self.split, "rb") as f:
            self.data = pickle.load(f)

        store_file = "./dataset/molecules/zinc_full/data_{}.bin".format(self.split)
        if self.num_graphs in [10000, 1000]:
            # loading the sampled indices from file ./zinc_molecules/<split>.index
            with open(data_dir + "/%s.index" % self.split, "r") as f:
                data_idx = [list(map(int, idx)) for idx in csv.reader(f)]
                self.data = [self.data[i] for i in data_idx[0]]

            assert len(self.data) == num_graphs, "Sample num_graphs again; available idx: train/val/test => 10k/1k/1k"
            store_file = "./dataset/molecules/zinc_full/data_subset_{}.bin".format(self.split)

        self.graph_lists = []
        self.graph_labels = []
        self.n_samples = len(self.data)
        if not os.path.isfile(store_file):
            self._prepare()
            save_graphs(store_file, self.graph_lists, {'glabel': torch.stack(self.graph_labels)})
        else:
            self.graph_lists, self.graph_labels = load_graphs(store_file)
            self.graph_labels = self.graph_labels['glabel']

        self.attachedNodeIds = 0

    def _prepare(self):
        print("preparing %d graphs for the %s set..." % (self.num_graphs, self.split.upper()))

        for molecule in tqdm(self.data):
            node_features = molecule['atom_type'].long()

            adj = molecule['bond_type']
            edge_list = (adj != 0).nonzero()  # converting adj matrix to edge_list

            edge_idxs_in_adj = edge_list.split(1, dim=1)
            edge_features = adj[edge_idxs_in_adj].reshape(-1).long()

            # Create the DGL Graph
            g = dgl.DGLGraph()
            g.add_nodes(molecule['num_atom'])
            g.ndata['feat'] = node_features

            for src, dst in edge_list:
                g.add_edges(src.item(), dst.item())
            g.edata['feat'] = edge_features

            self.graph_lists.append(g)
            self.graph_labels.append(molecule['logP_SA_cycle_normalized'])

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return len(self.graph_lists)
        # return self.n_samples

    def __getitem__(self, idx):
        """
            Get the (idx)-th sample.
            Parameters
            ---------
            idx : int
                The sample index.
            Returns
            -------
            (dgl.DGLGraph, int)
                DGLGraph with node feature stored in `feat` field
                And its label.
        """
        return self.graph_lists[idx], self.graph_labels[idx]

    def _prepare_full_graph(self):
        store_file = "./dataset/molecules/zinc_full/data_gfull_{}.bin".format(self.split)
        if not os.path.isfile(store_file):
            self.graph_lists = [make_full_graph(g) for g in self.graph_lists]
            save_graphs(store_file, self.graph_lists, {'glabel': self.graph_labels})
        else:
            self.graph_lists, self.graph_labels = load_graphs(store_file)
            self.graph_labels = self.graph_labels['glabel']

    def _get_kneighborhood_information(self, g, cutoff):
        tmp = torch.zeros((g.number_of_nodes(), g.number_of_nodes()), dtype=torch.int32)
        shortest_paths = dict(nx.all_pairs_shortest_path_length(dgl.to_networkx(g), cutoff=cutoff))
        for start_node, values in shortest_paths.items():
            shortest_dist = torch.zeros((g.number_of_nodes()))
            for target_node, dist in values.items():
                shortest_dist[target_node] = dist
            tmp[start_node, :] = shortest_dist
        tmp[tmp == 0] = -1
        g.ndata['nn-dist'] = tmp
        return g

    def _laplacian_positional_encoding(self, g, pos_enc_dim):
        """
            Graph positional encoding w/ Laplacian eigenvectors
        """
        # Laplacian
        A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
        N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
        L = sp.eye(g.number_of_nodes()) - N * A * N

        # Eigenvectors with numpy
        eigenvalues, eigenvectors = np.linalg.eig(L.toarray())
        idx = eigenvalues.argsort()  # increasing order
        eigenvalues, eigenvectors = eigenvalues[idx], np.real(eigenvectors[:, idx])
        g.ndata['lap_pos_enc'] = torch.from_numpy(eigenvectors[:, 1:pos_enc_dim + 1]).float()

        return g

    def prepare_ppr_graph(self, alphas):
        store_file = "./dataset/molecules/zinc_full/data_ppr_{}.bin".format(self.split)
        if not os.path.isfile(store_file):
            self.graph_lists = [self._compute_ppr(g, alphas) for g in tqdm(self.graph_lists)]
            save_graphs(store_file, self.graph_lists, {'glabel': self.graph_labels})
        else:
            self.graph_lists, self.graph_labels = load_graphs(store_file)
            self.graph_labels = self.graph_labels['glabel']

    def _compute_ppr(self, g, alphas):
        in_g = g
        g = nx.Graph(dgl.to_networkx(g))
        complete_ppr_dict = {}
        for alpha in tqdm(alphas):
            ppr_alpha = {}
            for node_id in range(g.number_of_nodes()):
                ppr = nx.pagerank(g,
                                  personalization={nid: 0 if nid != node_id else 1 for nid in range(g.number_of_nodes())},
                                  alpha=alpha,
                                  max_iter=600)
                ppr_alpha[node_id] = torch.tensor(list(ppr.values()))
            complete_ppr_dict[alpha] = ppr_alpha

            in_g.ndata['alpha_{}'.format(alpha)] = torch.tensor(torch.vstack(list(ppr_alpha.values())))

        return in_g

    def _compute_pagerank(self, g):
        full_g = dgl.from_networkx(nx.complete_graph(g.number_of_nodes()))
        real_ids = torch.nonzero(g.edata['real']).squeeze()
        fake_ids = torch.nonzero(g.edata['real'] == 0).squeeze()

        pos_g = dgl.edge_subgraph(full_g, real_ids)
        pos_g_nx = dgl.to_networkx(pos_g)
        pos_g_nx = nx.Graph(pos_g_nx)

        pr_pos_g = nx.pagerank(pos_g_nx, alpha=0.85)

        neg_g = dgl.edge_subgraph(full_g, fake_ids)
        neg_g_nx = dgl.to_networkx(neg_g)
        neg_g_nx = nx.Graph(neg_g_nx)
        pr_neg_g = nx.pagerank(neg_g_nx, alpha=0.85)

        g.ndata['pr_pos_g'] = torch.tensor(list(pr_pos_g.values()))
        g.ndata['pr_neg_g'] = torch.tensor(list(pr_neg_g.values()))

        return g

    def _addkhops(self, g, k_cutoff=2):
        paths = dict(nx.all_pairs_shortest_path(nx.Graph(dgl.to_networkx(g)), cutoff=k_cutoff))
        # remove self-loops
        paths = {key: {inner_key: path for inner_key, path in v.items() if inner_key != key} for key, v in paths.items()}
        twohop_g = nx.Graph(paths)
        full_g = dgl.from_networkx(twohop_g)

        # Here we copy over the node feature data and laplace encodings
        full_g.ndata['feat'] = g.ndata['feat']

        try:
            full_g.ndata['EigVecs'] = g.ndata['EigVecs']
            full_g.ndata['EigVals'] = g.ndata['EigVals']
        except:
            pass

        # Populate edge features w/ 0s
        full_g.edata['feat'] = torch.zeros(full_g.number_of_edges(), dtype=torch.long)
        full_g.edata['real'] = torch.zeros(full_g.number_of_edges(), dtype=torch.long)

        # Copy real edge data over
        full_g.edges[g.edges(form='uv')[0].tolist(), g.edges(form='uv')[1].tolist()].data['feat'] = g.edata['feat']
        full_g.edges[g.edges(form='uv')[0].tolist(), g.edges(form='uv')[1].tolist()].data['real'] = \
            torch.ones(g.edata['feat'].shape[0], dtype=torch.long)

        return full_g


class MoleculeDatasetDGL(torch.utils.data.Dataset):
    def __init__(self, name='ZINC', shell_attention=False, ppr_flag=False):
        t0 = time.time()
        self.name = name

        self.num_atom_type = 28  # known meta-info about the zinc dataset; can be calculated as well
        self.num_bond_type = 4  # known meta-info about the zinc dataset; can be calculated as well
        self.num_entities = self.num_atom_type
        self.num_rels = self.num_bond_type
        self.ppr_flag = ppr_flag
        self.enable_shell_attention = shell_attention

        data_dir = './dataset/molecules/zinc_full'
        if self.name == 'ZINC-full':
            data_dir = './dataset/molecules/zinc_full'
            self.train = MoleculeDGL(data_dir, 'train', num_graphs=220011)
            self.valid = MoleculeDGL(data_dir, 'val', num_graphs=24445)
            self.test = MoleculeDGL(data_dir, 'test', num_graphs=5000)
        else:
            self.train = MoleculeDGL(data_dir, 'train', num_graphs=10000)
            self.valid = MoleculeDGL(data_dir, 'val', num_graphs=1000)
            self.test = MoleculeDGL(data_dir, 'test', num_graphs=1000)
        print("Loaded ZINC - Time taken: {:.4f}s".format(time.time() - t0))

    def collate_dgl(self, samples):
        # TODO: check y stacking
        """ Code as explained in API dgl"""
        graphs, labels = map(list, zip(*samples))
        if self.ppr_flag:
            max_size = max([graphs[i].num_nodes() for i in range(0, len(graphs))])
            for graph_idx in range(0, len(graphs)):
                for alpha in self.ppr_alphas:
                    graphs[graph_idx].ndata['alpha_{}'.format(alpha)] = F.pad(graphs[graph_idx].ndata['alpha_{}'.format(alpha)],
                                                                      pad=(0, max_size - graphs[graph_idx].num_nodes()))
        if self.enable_shell_attention:
            # max_size = max([graphs[i].ndata['nn-dist'].shape[1] for i in range(0, len(graphs))])
            cumsum = 0
            for graph_idx in range(0, len(graphs)):
                # gather information of nodes having j distance to the query nodes (=nodes within a graph)
                nn_dist_tmp = [torch.where(graphs[graph_idx].ndata['nn-dist']==j) for j in range(1, self.cutoff+1)]
                # count number of nodes having a j distance
                bincounts = [torch.bincount(nn_dist_tmp[j][0]) for j in range(0, len(nn_dist_tmp))]
                # padding as number of nodes might differ (need for stacking in dgl; otherwise batching not possible)
                bincounts = [F.pad(bincounts[j], pad=(0, graphs[graph_idx].num_nodes() - bincounts[j].shape[0]), value=0) for j in range(0, len(bincounts))]
                # in a batch the nodes are re-index, hence, take care of this re-indexing by adding the number of nodes
                # seen so far in the batch
                nn_dist_tmp = [(nn_dist_tmp[j][0], nn_dist_tmp[j][1] + cumsum) for j in range(0, len(nn_dist_tmp))]
                # split the re-indexed target nodes according to the bincounts having observed above
                splits = [torch.split(nn_dist_tmp[j][1], bincounts[j].tolist()) for j in range(0, len(nn_dist_tmp))]
                # cumulated split information; GNNs gather information along k-hop distance, hence, we include the
                # information UP to the k-shell
                splits = [tuple((torch.cat([splits[k][node_id] for k in range(0, j)]))
                            for node_id in range(0, graphs[graph_idx].num_nodes()))
                            for j in range(1, len(splits)+1)]
                # padding such that the information can be batched in dgl
                padded_splits = [F.pad(torch.tensor(splits[j][node_id]),
                                       pad=(0, 50-splits[j][node_id].shape[0]),
                                       value=-1)
                                 for node_id in range(0, graphs[graph_idx].num_nodes())
                                 for j in range(0, len(splits))]
                # stack the information
                stacked_splits = torch.stack(padded_splits).view(graphs[graph_idx].num_nodes(), self.cutoff, -1)# .transpose(1,0)
                # store the information as node data in the graph
                graphs[graph_idx].ndata['nn-dist'] = stacked_splits
                # accumulate the number of nodes
                cumsum += graphs[graph_idx].num_nodes()

        # batched_graph = dgl.batch([entry for gs in graphs for entry in gs])
        batched_graph = dgl.batch(graphs)
        labels = torch.stack(labels)
        return batched_graph, labels

    def _make_full_graph(self):
        print("Full Graph: Start pre-processing graphs/Loading...")
        print("train set ...")
        self.train._prepare_full_graph()
        print("valid set ...")
        self.valid._prepare_full_graph()
        print("test set ...")
        self.test._prepare_full_graph()
        print("full graph-processing finished")

    def _add_kneighborhood_information(self, cutoff):
        print("Adding k neighborhood information...")
        print("train set ...")
        self.cutoff = cutoff-1
        self.max_node_size = max([g.number_of_nodes() for g in self.train.graph_lists])
        self.train.graph_lists = [self.train._get_kneighborhood_information(g, cutoff) for g in tqdm(self.train.graph_lists)]
        print("valid set ...")
        self.valid.graph_lists = [self.valid._get_kneighborhood_information(g, cutoff) for g in tqdm(self.valid.graph_lists)]
        print("test set ...")
        self.test.graph_lists = [self.test._get_kneighborhood_information(g, cutoff) for g in tqdm(self.test.graph_lists)]
        print("Adding k neighborhood information finished.")

    def _add_kneighorhood_graphs(self, cutoff):
        print("Creating k hop graphs...")
        print("train set ...")
        self.cutoff = cutoff
        self.train.graph_lists = [self.train._create_khop_graphs(g, cutoff) for g in tqdm(self.train.graph_lists[:100])]
        print("valid set ...")
        self.valid.graph_lists = [self.valid._create_khop_graphs(g, cutoff) for g in tqdm(self.valid.graph_lists[:100])]
        print("test set ...")
        self.test.graph_lists = [self.test._create_khop_graphs(g, cutoff) for g in tqdm(self.test.graph_lists[:100])]
        print("Created k hop graphs.")

    def _add_laplacian_positional_encodings(self, pos_enc_dim):
        # Graph positional encoding v/ Laplacian eigenvectors
        print("Laplacian Positional Encoding: Start pre-processing graphs...")
        print("train set ...")
        self.train.graph_lists = [self.train._laplacian_positional_encoding(g, pos_enc_dim)
                                  for g in tqdm(self.train.graph_lists)]
        print("valid set ...")
        self.valid.graph_lists = [self.valid._laplacian_positional_encoding(g, pos_enc_dim)
                                  for g in tqdm(self.valid.graph_lists)]
        print("test set ...")
        self.test.graph_lists = [self.test._laplacian_positional_encoding(g, pos_enc_dim)
                                 for g in tqdm(self.test.graph_lists)]
        print("LPE processing finished.")

    def _addkhops(self, k_cutoff=2):
        print("Add {} hops: start pre-processing graphs ...".format(k_cutoff))
        print("train set ...")
        self.train.graph_lists = [self.train._addkhops(g, k_cutoff) for g in tqdm(self.train.graph_lists)]
        print("valid set ...")
        self.valid.graph_lists = [self.valid._addkhops(g, k_cutoff) for g in tqdm(self.valid.graph_lists)]
        print("test set ...")
        self.test.graph_lists = [self.test._addkhops(g, k_cutoff) for g in tqdm(self.test.graph_lists)]
        print("{}-hops successfully added".format(k_cutoff))


def make_full_graph(g):
    full_g = dgl.from_networkx(nx.complete_graph(g.number_of_nodes()))

    # Here we copy over the node feature data and laplace encodings
    full_g.ndata['feat'] = g.ndata['feat']

    try:
        full_g.ndata['EigVecs'] = g.ndata['EigVecs']
        full_g.ndata['EigVals'] = g.ndata['EigVals']
    except:
        pass

    # Populate edge features w/ 0s
    full_g.edata['feat'] = torch.zeros(full_g.number_of_edges(), dtype=torch.long)
    full_g.edata['real'] = torch.zeros(full_g.number_of_edges(), dtype=torch.long)

    # Copy real edge data over
    full_g.edges[g.edges(form='uv')[0].tolist(), g.edges(form='uv')[1].tolist()].data['feat'] = g.edata['feat']
    full_g.edges[g.edges(form='uv')[0].tolist(), g.edges(form='uv')[1].tolist()].data['real'] = \
        torch.ones(g.edata['feat'].shape[0], dtype=torch.long)

    return full_g


class load_SBMsDataSetDGL(torch.utils.data.Dataset):

    def __init__(self, data_dir, name, split):

        self.split = split
        self.is_test = split.lower() in ['test', 'val']
        with open(os.path.join(data_dir, "SBM_" + name + '_%s.pkl' % self.split), 'rb') as f:
            self.dataset = pickle.load(f)
        self.node_labels = []
        self.graph_lists = []
        self.n_samples = len(self.dataset)

        if not os.path.isfile("./dataset/SBMs/{}/data_{}.bin".format(name, self.split)):
            self._prepare()
            save_graphs("./dataset/SBMs/{}/data_{}.bin".format(name, self.split),
                        self.graph_lists,
                        {'nlabel_{}'.format(idx): self.node_labels[idx] for idx in range(0, len(self.node_labels))}
                        )
        else:
            self.graph_lists, self.node_labels = load_graphs("./dataset/SBMs/{}/data_{}.bin".format(name, self.split))
            self.node_labels = [self.node_labels['nlabel_{}'.format(idx)] for idx in range(0, len(self.node_labels))]

    def _prepare(self):
        print("preparing %d graphs for the %s set..." % (self.n_samples, self.split.upper()))

        for data in tqdm(self.dataset):

            node_features = data.node_feat
            edge_list = (data.W != 0).nonzero()  # converting adj matrix to edge_list

            # Create the DGL Graph
            g = dgl.DGLGraph()
            g.add_nodes(node_features.size(0))
            g.ndata['feat'] = node_features.long()
            for src, dst in edge_list:
                g.add_edges(src.item(), dst.item())

            # adding edge features for Residual Gated ConvNet
            # edge_feat_dim = g.ndata['feat'].size(1) # dim same as node feature dim
            edge_feat_dim = 1  # dim same as node feature dim
            g.edata['feat'] = torch.ones(g.number_of_edges(), edge_feat_dim)

            self.graph_lists.append(g)
            self.node_labels.append(data.node_label)

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return len(self.graph_lists)
        # return self.n_samples

    def __getitem__(self, idx):
        """
            Get the idx^th sample.
            Parameters
            ---------
            idx : int
                The sample index.
            Returns
            -------
            (dgl.DGLGraph, int)
                DGLGraph with node feature stored in `feat` field
                And its label.
        """
        return self.graph_lists[idx], self.node_labels[idx]

    def _laplacian_positional_encoding(self, g, pos_enc_dim):
        """
            Graph positional encoding v/ Laplacian eigenvectors
        """

        # Laplacian
        A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
        N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
        L = sp.eye(g.number_of_nodes()) - N * A * N

        # Eigenvectors with numpy
        eigenvalues, eigenvectors = np.linalg.eig(L.toarray())
        idx = eigenvalues.argsort()  # increasing order
        eigenvalues, eigenvectors = eigenvalues[idx], np.real(eigenvectors[:, idx])
        g.ndata['lap_pos_enc'] = torch.from_numpy(eigenvectors[:, 1:pos_enc_dim + 1]).float()

        return g

    def _addkhops(self, g, k_cutoff=2):
        paths = dict(nx.all_pairs_shortest_path(nx.Graph(dgl.to_networkx(g)), cutoff=k_cutoff))
        # remove self-loops
        paths = {key: {inner_key: path
                       for inner_key, path in v.items()
                       if inner_key != key} for key, v in paths.items()}
        twohop_g = nx.Graph(paths)
        full_g = dgl.from_networkx(twohop_g)

        # Here we copy over the node feature data and laplace encodings
        full_g.ndata['feat'] = g.ndata['feat']

        try:
            full_g.ndata['EigVecs'] = g.ndata['EigVecs']
            full_g.ndata['EigVals'] = g.ndata['EigVals']
        except:
            pass

        # Populate edge features w/ 0s
        full_g.edata['feat'] = torch.zeros(full_g.number_of_edges(), dtype=torch.long)
        full_g.edata['real'] = torch.zeros(full_g.number_of_edges(), dtype=torch.long)

        # Copy real edge data over
        full_g.edges[g.edges(form='uv')[0].tolist(), g.edges(form='uv')[1].tolist()].data['feat'] = \
            torch.ones(g.edata['feat'].shape[0], dtype=torch.long)
        full_g.edges[g.edges(form='uv')[0].tolist(), g.edges(form='uv')[1].tolist()].data['real'] = \
            torch.ones(g.edata['feat'].shape[0], dtype=torch.long)

        return full_g


class SBMsDatasetDGL(torch.utils.data.Dataset):

    def __init__(self, name):
        """
            TODO
        """
        start = time.time()
        print("[I] Loading data ...")
        self.name = name
        data_dir = os.path.join('dataset/SBMs', self.name)
        self.train = load_SBMsDataSetDGL(data_dir, name, split='train')
        self.test = load_SBMsDataSetDGL(data_dir, name, split='test')
        self.valid = load_SBMsDataSetDGL(data_dir, name, split='val')
        print("[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time() - start))

        self.num_entities = torch.unique(self.train[0][0].ndata['feat'], dim=0).size(0)
        self.num_rels=2

    # form a mini batch from a given list of samples = [(graph, label) pairs]
    def collate_dgl(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        labels = torch.cat(labels).long()
        batched_graph = dgl.batch(graphs)
        return batched_graph, labels

    def _addkhops(self, k_cutoff=2):
        print("Add {} hops: start pre-processing graphs ...".format(k_cutoff))
        print("train set ...")
        store_file = "./dataset/SBMs/{}/data_{}hop_{}.bin".format(self.name, k_cutoff, "train")
        if not os.path.isfile(store_file):
            self.train.graph_lists = [self.train._addkhops(g) for g in tqdm(self.train.graph_lists)]
            save_graphs("./dataset/SBMs/{}/data_{}hop_{}.bin".format(self.name, k_cutoff, "train"),
                        self.train.graph_lists,
                        {'nlabel_{}'.format(idx): self.train.node_labels[idx] for idx in range(0, len(self.train.node_labels))}
                        )
        else:
            self.train.graph_lists, node_labels = load_graphs("./dataset/SBMs/{}/data_{}hop_{}.bin".format(self.name, k_cutoff, "train"))
            self.train.node_labels = [node_labels['nlabel_{}'.format(idx)] for idx in range(0, len(node_labels))]

        print("valid set ...")
        store_file = "./dataset/SBMs/{}/data_{}hop_{}.bin".format(self.name, k_cutoff, "valid")
        if not os.path.isfile(store_file):
            self.valid.graph_lists = [self.valid._addkhops(g) for g in tqdm(self.valid.graph_lists)]
            save_graphs("./dataset/SBMs/{}/data_{}hop_{}.bin".format(self.name, k_cutoff, "valid"),
                        self.valid.graph_lists,
                        {'nlabel_{}'.format(idx): self.valid.node_labels[idx] for idx in range(0, len(self.valid.node_labels))}
                        )
        else:
            self.valid.graph_lists, node_labels = load_graphs("./dataset/SBMs/{}/data_{}hop_{}.bin".format(self.name, k_cutoff, "valid"))
            self.valid.node_labels = [node_labels['nlabel_{}'.format(idx)] for idx in range(0, len(node_labels))]
        # self.valid.graph_lists = [self.valid._addkhops(g) for g in self.valid.graph_lists]

        print("test set ...")
        store_file = "./dataset/SBMs/{}/data_{}hop_{}.bin".format(self.name, k_cutoff, "test")
        if not os.path.isfile(store_file):
            self.test.graph_lists = [self.test._addkhops(g) for g in tqdm(self.test.graph_lists)]
            save_graphs("./dataset/SBMs/{}/data_{}hop_{}.bin".format(self.name, k_cutoff, "test"),
                        self.test.graph_lists,
                        {'nlabel_{}'.format(idx): self.test.node_labels[idx] for idx in range(0, len(self.test.node_labels))}
                        )
        else:
            self.test.graph_lists, node_labels = load_graphs("./dataset/SBMs/{}/data_{}hop_{}.bin".format(self.name, k_cutoff, "test"))
            self.test.node_labels = [node_labels['nlabel_{}'.format(idx)] for idx in range(0, len(node_labels))]
        # self.test.graph_lists = [self.test._addkhops(g) for g in self.test.graph_lists]
        print("{}-hops successfully added".format(k_cutoff))

    def _add_laplacian_positional_encodings(self, pos_enc_dim):
        # Graph positional encoding v/ Laplacian eigenvectors
        print("Laplacian Positional Encoding: Start pre-processing graphs...")
        print("train set ...")
        self.train.graph_lists = [self.train._laplacian_positional_encoding(g, pos_enc_dim)
                                  for g in tqdm(self.train.graph_lists)]
        print("valid set ...")
        self.valid.graph_lists = [self.valid._laplacian_positional_encoding(g, pos_enc_dim)
                                  for g in tqdm(self.valid.graph_lists)]
        print("test set ...")
        self.test.graph_lists = [self.test._laplacian_positional_encoding(g, pos_enc_dim)
                                 for g in tqdm(self.test.graph_lists)]
        print("LPE processing finished")
