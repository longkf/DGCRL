import networkx as nx
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch
from torch_geometric.utils import coalesce, scatter, degree
import copy
from torch_geometric.utils import negative_sampling
import numpy as np
import torch.nn.functional as F


def show_tsne(x, edge_label):
    tsne = TSNE(n_components=2, learning_rate=100).fit_transform(x)
    nums_tp = int(torch.sum(edge_label))
    tp_edges = tsne[:nums_tp, :]
    tf_edges = tsne[nums_tp:, :]
    plt.figure()
    plt.scatter(tp_edges[:, 0], tp_edges[:, 1], c='r', s=5)
    plt.scatter(tf_edges[:, 0], tf_edges[:, 1], c='b', s=5)
    plt.show()


def get_edge_feature(x, edge_index):
    s, t = edge_index
    line_x = torch.cat([x[s], x[t], x[s] + x[t]], dim=1)
    return line_x


def graph2line_graph(edge_index, num_nodes):
    edge_index, edge_attr = coalesce(edge_index, None, num_nodes=num_nodes)
    row, col = edge_index
    i = torch.arange(row.size(0), dtype=torch.long, device=row.device)
    count = scatter(torch.ones_like(row), row, dim=0,
                    dim_size=num_nodes, reduce='sum')
    cumsum = torch.cat([count.new_zeros(1), count.cumsum(0)], dim=0)
    cols = [
        i[cumsum[col[j]]:cumsum[col[j] + 1]]
        for j in range(col.size(0))
    ]
    rows = [row.new_full((c.numel(),), j) for j, c in enumerate(cols)]
    row, col = torch.cat(rows, dim=0), torch.cat(cols, dim=0)
    edge_index = torch.stack([row, col], dim=0)
    return edge_index


def dropout_feature(x, drop_prob):
    x = copy.deepcopy(x.detach())
    rand = torch.rand(x.size(1))
    mask = torch.where(rand < drop_prob, 1, 0)
    x[:, mask] = 0
    return x * mask


def getAdjMatrix(edge_index, nums_node):
    A = torch.zeros(size=[nums_node * nums_node])
    move = torch.tensor([nums_node, 1], dtype=torch.int64)
    move = move @ edge_index.cpu()
    A[move] = 1
    A = torch.reshape(A, (nums_node, nums_node)).to(edge_index.device)
    return A


def row_normalize_adj(edge_index, nums_node):
    adj = getAdjMatrix(edge_index, nums_node)
    degree = torch.sum(adj, dim=1, keepdim=True)
    norm_adj = torch.nan_to_num(adj / degree, nan=0)
    return norm_adj


def showGraph(num_nodes, edge_index):
    G = nx.DiGraph()
    G.add_nodes_from([i for i in range(num_nodes)])
    edges = np.array(edge_index.t().contiguous(), dtype=int)
    G.add_edges_from(edges)
    adj = nx.adjacency_matrix(G)
    adj = adj.todense()
    print(adj)
    plt.figure(figsize=(6, 4))
    pos = nx.circular_layout(G)
    nx.draw_networkx(G, pos=pos, with_labels=True)
    plt.show()


def set_random_zeros_by_proportion(X, proportion):
    N, F = X.shape
    zero_count = int(N * F * proportion)
    indices = torch.randperm(N * F)[:zero_count]
    X_flat = X.flatten()
    X_flat[indices] = 0
    X_modified = X_flat.view(N, F)
    return X_modified


def add_gaussian_noise(X, sigma):
    noise = torch.randn_like(X) * sigma
    Y = X + noise
    return Y
