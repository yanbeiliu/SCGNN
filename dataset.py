from dgl.data import CoraGraphDataset, CitationGraphDataset
from sklearn.preprocessing import MinMaxScaler
import scipy.sparse as sp
import networkx as nx
import numpy as np
import os
from matplotlib import pyplot as plt
from scipy.linalg import fractional_matrix_power, inv


def download(dataset):
    if dataset == 'cora':
        return CoraGraphDataset()
    elif dataset == 'citeseer' or 'pubmed':
        return CitationGraphDataset(name=dataset)
    else:
        return None


def load(dataset):
    datadir = os.path.join('data', dataset)

    if not os.path.exists(datadir):

        os.makedirs(datadir)
        ds = download(dataset)
        adj = nx.to_numpy_array(ds.graph)
        diff = compute_ppr(ds.graph, 0.2)
        feat = ds.features[:]
        labels = ds.labels[:]
        idx_train = np.argwhere(ds.train_mask == 1).reshape(-1)
        idx_val = np.argwhere(ds.val_mask == 1).reshape(-1)
        idx_test = np.argwhere(ds.test_mask == 1).reshape(-1)

        np.save(f'{datadir}/adj.npy', adj)
        np.save(f'{datadir}/diff.npy', diff)
        np.save(f'{datadir}/feat.npy', feat)
        np.save(f'{datadir}/labels.npy', labels)
        np.save(f'{datadir}/idx_train.npy', idx_train)
        np.save(f'{datadir}/idx_val.npy', idx_val)
        np.save(f'{datadir}/idx_test.npy', idx_test)
    else:

        adj = np.load(f'{datadir}/adj.npy')
        diff = np.load(f'{datadir}/diff.npy')
        feat = np.load(f'{datadir}/feat.npy')
        labels = np.load(f'{datadir}/labels.npy')
        idx_train = np.load(f'{datadir}/idx_train.npy')
        idx_val = np.load(f'{datadir}/idx_val.npy')
        idx_test = np.load(f'{datadir}/idx_test.npy')

    if dataset == 'citeseer':
        feat = preprocess_features(feat)

        epsilons = [1e-5, 1e-4, 1e-3, 1e-2]
        avg_degree = np.sum(adj) / adj.shape[0]
        epsilon = epsilons[np.argmin([abs(avg_degree - np.argwhere(diff >= e).shape[0] / diff.shape[0])
                                      for e in epsilons])]

        diff[diff < epsilon] = 0.0
        scaler = MinMaxScaler()
        scaler.fit(diff)
        diff = scaler.transform(diff)

    # 此上，adj, diff, feat, labels, idx_train, idx_val, idx_test全是ndarray类型,feature已经归一化了
    adj = normalize_adj(adj + sp.eye(adj.shape[0])).todense()
    return adj, diff, feat, labels, idx_train, idx_val, idx_test


def compute_ppr(graph: nx.Graph, alpha=0.2, self_loop=True):
    a = nx.convert_matrix.to_numpy_array(graph)
    if self_loop:
        a = a + np.eye(a.shape[0])                               # A^ = A + I_n
    d = np.diag(np.sum(a, 1))                                    # D^ = Sigma A^_ii
    dinv = fractional_matrix_power(d, -0.5)                      # D^(-1/2)
    at = np.matmul(np.matmul(dinv, a), dinv)                     # A~ = D^(-1/2) x A^ x D^(-1/2)
    return alpha * inv((np.eye(a.shape[0]) - (1 - alpha) * at))  # a(I_n-(1-a)A~)^-1


def compute_heat(graph: nx.Graph, t=5, self_loop=True):
    a = nx.convert_matrix.to_numpy_array(graph)
    if self_loop:
        a = a + np.eye(a.shape[0])
    d = np.diag(np.sum(a, 1))
    return np.exp(t * (np.matmul(a, inv(d)) - 1))


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    # 标准化特征矩阵并转换为元组表示
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    if isinstance(features, np.ndarray):
        return features
    else:
        return features.todense(), sparse_to_tuple(features)


def normalize_adj(adj, self_loop=True):
    """Symmetrically normalize adjacency matrix."""
    if self_loop:
        adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


if __name__ == '__main__':
    load('cora')