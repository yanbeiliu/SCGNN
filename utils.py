# 此文件是定义了一些需要的工具函数
import numpy as np  # 导入numpy包，且用np名称等价
import scipy.sparse as sp  # scipy.sparse稀疏矩阵包，且用sp名称等价
import torch
from scipy.linalg import fractional_matrix_power, inv


'''
先将所有由字符串表示的标签数组用set保存，set的重要特征就是元素没有重复，
因此表示成set后可以直接得到所有标签的总数，随后为每个标签分配一个编号，创建一个单位矩阵，
单位矩阵的每一行对应一个one-hot向量，也就是np.identity(len(classes))[i, :]，
再将每个数据对应的标签表示成的one-hot向量，类型为numpy数组
'''


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}

    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)

    return labels_onehot


def load_data(path="../data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))
    #  content file的读取
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    # 论文样本的独自信息的数组
    # np.genfromtxt()函数用于从.csv文件或.tsv文件中生成数组
    # np.genfromtxt(fname, dtype)
    # frame：文件名	../data/cora/cora.content		dtype：数据类型	str字符串
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # 提取样本的特征，并将其转换为csr矩阵（压缩稀疏行矩阵），用行索引、列索引和值表示矩阵
    # [:, 1:-1]是指行全部选中、列选取第二列至倒数第二列，float32类型
    # 这句功能就是去除论文样本的编号和类别，留下每篇论文的词向量，并将稀疏矩阵编码压缩
    labels = encode_onehot(idx_features_labels[:, -1])
    # 提取论文样本的类别标签，并将其转换为one-hot编码形式

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # 提取论文样本的编号id数组
    idx_map = {j: i for i, j in enumerate(idx)}
    # 由样本id到样本索引的映射字典
    # enumerate()将可遍历对象组合成一个含数据下标和数据的索引序列
    # {}生成了字典，论文编号id作为索引的键，顺序数据下标值i作为键值:0,1,2,...
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    # 论文样本之间的引用关系的数组
    # np.genfromtxt()函数用于从.csv文件或.tsv文件中生成数组
    # np.genfromtxt(fname, dtype)
    # frame：文件名	../data/cora/cora.cites		dtype：数据类型	int32
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    # 将论文样本之间的引用关系用样本字典索引之间的关系表示，
    # 说白了就是将论文引用关系数组中的数据(论文id）替换成对应字典的索引键值
    # list()用于将元组转换为列表。flatten()是将关系数组降为一维，默认按一行一行排列
    # map()是对降维后的一维关系数组序列中的每一个元素调用idx_map.get进行字典索引，
    # 即将一维的论文引用关系数组中论文id转化为对应的键值数据
    # .shape是读取数组的维度，.reshape()是将一维数组复原成原来维数形式

    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    # 构建图的邻接矩阵，用坐标形式的稀疏矩阵表示，非对称邻接矩阵
    # edges.shape[0]表示引用关系数组的维度数（行数），np.ones全1的n维数组
    # edges[:, 0]被引用论文的索引数组做行号row，edges[:, 1]引用论文的索引数组做列号col
    # labels.shape[0]总论文样本的数量，做方阵维数
    # 前面说白了就是引用论文的索引做列，被引用论文的索引做行，然后在这个矩阵面填充1，其余填充0

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)  # coo变csr
    # 将非对称邻接矩阵转变为对称邻接矩阵（有向图转无向图）
    # A.multiply(B)是A与B的Hadamard乘积，A>B是指按位将A大于B的位置进行置1其余置0（仅就这里而言可以这么理解，我没找到具体解释）
    # adj=adj+((adj转置)⊙(adj.T > adj))-((adj)⊙(adj.T > adj))
    # 基本上就是将非对称矩阵中的非0元素进行对应对称位置的填补，得到对称邻接矩阵

    b = np.array(adj.todense())

    features = normalize_features(features)  # csr
    # features是样本特征的压缩稀疏矩阵，行规范化稀疏矩阵，具体函数后面有定义

    adj = normalize_adj(adj + sp.eye(adj.shape[0]))   # D^-1/2（A+In）D^-1/2
    # adj = normalize_features(adj + sp.eye(adj.shape[0]))   # D^-1（A+In）normalize_features还是csr，normalize_adj就csr变csc

    idx_train = range(140)  # 0~139，训练集索引列表
    idx_val = range(200, 500)  # 200~499，验证集索引列表
    idx_test = range(500, 1500)  # 500~1499，测试集索引列表

    features = torch.FloatTensor(np.array(features.todense())[np.newaxis])

    labels = torch.LongTensor(np.where(labels)[1])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj,  b, features, labels, idx_train, idx_val, idx_test


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

'''
def normalize(mx):    # 这里是计算D^-1A，而不是计算论文中的D^-1/2AD^-1/2
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
'''


def accuracy(output, labels):     # 准确率，此函数可参考学习借鉴复用
    # max(1)返回每一行最大值组成的一维数组和索引,output.max(1)[1]表示最大值所在的索引indice
    # type_as()将张量转化为labels类型
    preds = output.max(1)[1].type_as(labels)
    # eq是判断preds与labels是否相等，相等的话对应元素置1，不等置0
    # 记录等于preds的label eq:equal
    correct = preds.eq(labels).double()
    correct = correct.sum()   # 对其求和，即求出相等(置1)的个数
    return correct / len(labels)


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
    a = a + np.eye(a.shape[0])                                    # A^ = A + I_n
    d = np.diag(np.sum(a, 1))                                     # D^ = Sigma A^_ii
    dinv = fractional_matrix_power(d, -0.5)                       # D^(-1/2)
    at = np.matmul(np.matmul(dinv, a), dinv)                      # A~ = D^(-1/2) x A^ x D^(-1/2)
    return alpha * inv((np.eye(a.shape[0]) - (1 - alpha) * at))   # a(I_n-(1-a)A~)^-1


def compute_heat(a, t=5):
    a = a + np.eye(a.shape[0])
    d = np.diag(np.sum(a, 1))
    return np.exp(t * (np.matmul(a, inv(d)) - 1))