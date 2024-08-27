import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import sys
import torch
import torch.nn as nn
from scipy.linalg import fractional_matrix_power, inv


def parse_skipgram(fname):
    with open(fname) as f:
        toks = list(f.read().split())  # split() 通过指定分隔符对字符串进行切片，如果参数 num 有指定值，则分隔 num+1 个子字符串
    nb_nodes = int(toks[0])  # list() 方法用于将元组转换为列表。
    nb_features = int(toks[1])
    ret = np.empty((nb_nodes, nb_features))  # empty(shape[, dtype, order])  依给定的shape, 和数据类型 dtype
    # 返回一个一维或者多维数组，数组的元素不为空，为随机产生的数据。
    it = 2
    for i in range(nb_nodes):
        cur_nd = int(toks[it]) - 1
        it += 1
        for j in range(nb_features):
            cur_ft = float(toks[it])
            ret[cur_nd][j] = cur_ft  # 将ret数组cur_nd，j的位置的值替换为cur_ft
            it += 1
    return ret


# Process a (subset of) a TU dataset into standard form  将TU数据集的（子集）处理为标准格式


def process_tu(data, nb_nodes):
    nb_graphs = len(data)
    ft_size = data.num_features

    features = np.zeros((nb_graphs, nb_nodes, ft_size))
    adjacency = np.zeros((nb_graphs, nb_nodes, nb_nodes))
    labels = np.zeros(nb_graphs)
    sizes = np.zeros(nb_graphs, dtype=np.int32)
    masks = np.zeros((nb_graphs, nb_nodes))

    for g in range(nb_graphs):
        sizes[g] = data[g].x.shape[0]
        features[g, :sizes[g]] = data[g].x
        labels[g] = data[g].y[0]
        masks[g, :sizes[g]] = 1.0
        e_ind = data[g].edge_index
        coo = sp.coo_matrix((np.ones(e_ind.shape[1]), (e_ind[0, :], e_ind[1, :])), shape=(nb_nodes, nb_nodes))
        adjacency[g] = coo.todense()

    return features, adjacency, labels, sizes, masks


def micro_f1(logits, labels):
    # Compute predictions  计算预测
    preds = torch.round(nn.Sigmoid()(logits))

    # Cast to avoid trouble
    preds = preds.long()
    labels = labels.long()

    # Count true positives, true negatives, false positives, false negatives  统计真阳性，真阴性，假阳性，假阴性
    tp = torch.nonzero(preds * labels).shape[0] * 1.0
    tn = torch.nonzero((preds - 1) * (labels - 1)).shape[0] * 1.0
    fp = torch.nonzero(preds * (labels - 1)).shape[0] * 1.0
    fn = torch.nonzero((preds - 1) * labels).shape[0] * 1.0

    # Compute micro-f1 score  计算micro-f1分数
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = (2 * prec * rec) / (prec + rec)
    return f1


"""
 Prepare adjacency matrix by expanding up to a given neighbourhood.
 This will insert loops on every node.
 Finally, the matrix is converted to bias vectors.
 Expected shape: [graph, nodes, nodes]
 通过向上扩展到给定的邻域来准备邻接矩阵。
 这将在每个节点上插入循环。
 最后，矩阵被转换成偏置向量。
 预期形状：[图形，节点，节点]
"""


def adj_to_bias(adj, sizes, nhood=1):
    nb_graphs = adj.shape[0]
    mt = np.empty(adj.shape)
    for g in range(nb_graphs):
        mt[g] = np.eye(adj.shape[1])
        for _ in range(nhood):
            mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
        for i in range(sizes[g]):
            for j in range(sizes[g]):
                if mt[g][i][j] > 0.0:
                    mt[g][i][j] = 1.0
    return -1e9 * (1.0 - mt)


###############################################
# This section of code adapted from tkipf/gcn #
###############################################

def parse_index_file(filename):  # 该函数说白了就是给论文id从0到2707排号
    """Parse index file."""  # 解析索引文件
    index = []
    for line in open(filename):
        index.append(int(line.strip()))  # strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str):  # {'pubmed', 'citeseer', 'cora'}
    """Load data."""
    print('Loading {} dataset...'.format(dataset_str))
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data_xy/{}/ind.{}.{}".format(dataset_str, dataset_str, names[i]),
                  'rb') as f:  # with语句来自动帮我们调用close()方法,
            # 文件读取不成功自动.close
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))  # .appand 是为object列表最后添加元素
            else:  # pkl.load 用来保存数据
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)  # tuple() 函数将列表转换为元组。
    # x：140*1433，为邻接矩阵用稀疏矩阵存储,y:140*7,为标签，
    # tx：1000*1433，为邻接矩阵用稀疏矩阵存储，ty:1000*7,为标签
    # allx:1708*1433,用稀疏矩阵存储,ally:1708*7,为标签

    test_idx_reorder = parse_index_file("data_xy/{}/ind.{}.test.index".format(dataset_str, dataset_str))
    # ind.cora.test.index => 测试实例的id，（1000，）,从1708到2707号

    test_idx_range = np.sort(test_idx_reorder)
    # np.sort()函数的作用是对给定的数组的元素进行排序
    # a：需要排序的数组
    # axis：指定按什么排序，默认axis = 1 按行排序， axis = 0 按列排序

    if dataset_str == 'citeseer':
        # 修复citeseer数据集（图中有一些孤立的节点）
        # 找到孤立的节点，将它们作为零向量添加到正确的位置
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()  # vstack()是对数组进行拼接, features 是2708*1433，应该是特征矩阵，用稀疏形式存储
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))  # adj是邻接矩阵，2708*2708

    b = np.array(adj.todense())

    labels = np.vstack((ally, ty))  # labels 2708*7,one-hot向量编码

    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    idx_test = test_idx_range.tolist()  # tolist() 将矩阵（matrix）和数组（array）转化为列表。
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)

    # train_mask = sample_mask(idx_train, labels.shape[0])  # [ True  True  True ... False False False] (2708,)
    # val_mask = sample_mask(idx_val, labels.shape[0])  # [False False False ... False False False] (2708,)
    # test_mask = sample_mask(idx_test, labels.shape[0])  # [False False False ...  True  True  True] (2708,)
    #
    # y_train = np.zeros(labels.shape)
    # y_val = np.zeros(labels.shape)
    # y_test = np.zeros(labels.shape)
    # y_train[train_mask, :] = labels[train_mask, :]  # (2708, 7)
    # y_val[val_mask, :] = labels[val_mask, :]  # (2708, 7)
    # y_test[test_mask, :] = labels[test_mask, :]  # (2708, 7)

    return adj, b, features, labels, idx_train, idx_val, idx_test


def sparse_to_tuple(sparse_mx, insert_batch=False):
    """Convert sparse matrix to tuple representation."""
    """Set insert_batch=True if you want to insert a batch dimension."""
    '''
    将稀疏矩阵转换为元组表示形式
    如果要插入批次维度，请将insert_batch=True。
    '''

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        if insert_batch:
            coords = np.vstack((np.zeros(mx.row.shape[0]), mx.row, mx.col)).transpose()
            values = mx.data
            shape = (1,) + mx.shape
        else:
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


def standardize_data(f, train_mask):
    """Standardize feature matrix and convert to tuple representation"""
    # 标准化特征矩阵并转换为元组表示
    # standardize data
    f = f.todense()
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = f[:, np.squeeze(np.array(sigma > 0))]
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = (f - mu) / sigma
    return f


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    # 行标准化特征矩阵并转换为元组表示
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)


def normalize_adj(adj):
    # 该函数就是对邻接矩阵左右都乘D^-1/2
    """Symmetrically normalize adjacency matrix."""
    # 邻接矩阵的对称正规化
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    # 简单GCN模型邻接矩阵的预处理及元组表示转换
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    # Scipy中的sparse matrix转换为PyTorch中的sparse matrix，此函数可参考学习借鉴复用
    # 构建稀疏张量，一般需要Coo索引、值以及形状大小等信息
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def compute_ppr(a, alpha=0.2):
    a = a + np.eye(a.shape[0])                                   # A^ = A + I_n
    d = np.diag(np.sum(a, 1))                                    # D^ = Sigma A^_ii
    dinv = fractional_matrix_power(d, -0.5)                      # D^(-1/2)
    at = np.matmul(np.matmul(dinv, a), dinv)                     # A~ = D^(-1/2) x A^ x D^(-1/2)
    return alpha * inv((np.eye(a.shape[0]) - (1 - alpha) * at))  # a(I_n-(1-a)A~)^-1


def compute_heat(a, t=5):
    a = a + np.eye(a.shape[0])
    d = np.diag(np.sum(a, 1))
    return np.exp(t * (np.matmul(a, inv(d)) - 1))


# def set_train_val_test_split(
#         seed: int,
#         data: Data,
#         num_development: int = 1500,
#         num_per_class: int = 20) -> Data:
#     development_seed = 1684992425
#     rnd_state = np.random.RandomState(development_seed)
#     num_nodes = data.y.shape[0]
#     development_idx = rnd_state.choice(num_nodes, num_development, replace=False)
#     test_idx = [i for i in np.arange(num_nodes) if i not in development_idx]
#
#     train_idx = []
#     rnd_state = np.random.RandomState(seed)
#     for c in range(data.y.max() + 1):
#         class_idx = development_idx[np.where(data.y[development_idx].cpu() == c)[0]]
#         train_idx.extend(rnd_state.choice(class_idx, num_per_class, replace=False))
#
#     val_idx = [i for i in development_idx if i not in train_idx]
#
#     def get_mask(idx):
#         mask = torch.zeros(num_nodes, dtype=torch.bool)
#         mask[idx] = 1
#         return mask
#
#     data.train_mask = get_mask(train_idx)
#     data.val_mask = get_mask(val_idx)
#     data.test_mask = get_mask(test_idx)
#
#     return data