from pygcn.utils import encode_onehot
from pygcn.utils import sparse_mx_to_torch_sparse_tensor
from pygcn.utils import normalize_adj, normalize_features, compute_ppr
import time
import torch.nn.functional as F
import torch.optim as optim
from pygcn.utils import load_data, accuracy
from pygcn.models import GCN

# 此文件是定义了一些需要的工具函数
import numpy as np  # 导入numpy包，且用np名称等价
import scipy.sparse as sp  # scipy.sparse稀疏矩阵包，且用sp名称等价
import torch
'''
labels的onehot编码，前后结果对比
'''
# 读取原始数据集
path="D:/pycharm/pythonProject/cora/"
dataset = "cora"
idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))

RawLabels=idx_features_labels[:, -1]
# print("原始论文类别（label）：\n",RawLabels)
# ['Neural_Networks' 'Rule_Learning' 'Reinforcement_Learning' ...
# 'Genetic_Algorithms' 'Case_Based' 'Neural_Networks']
# print(len(RawLabels))       # 2708

classes = set(RawLabels)       # set() 函数创建一个无序不重复元素集
# print("原始标签的无序不重复元素集\n", classes)
# {'Genetic_Algorithms', 'Probabilistic_Methods', 'Reinforcement_Learning', 'Neural_Networks', 'Theory', 'Case_Based', 'Rule_Learning'}


# enumerate()函数生成序列，带有索引i和值c。
# 这一句将string类型的label变为onehot编码的label，建立映射关系
classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
# print("原始标签与onehot编码结果的映射字典\n",classes_dict)
#  {'Genetic_Algorithms': array([1., 0., 0., 0., 0., 0., 0.]), 'Probabilistic_Methods': array([0., 1., 0., 0., 0., 0., 0.]),
#   'Reinforcement_Learning': array([0., 0., 1., 0., 0., 0., 0.]), 'Neural_Networks': array([0., 0., 0., 1., 0., 0., 0.]),
#   'Theory': array([0., 0., 0., 0., 1., 0., 0.]), 'Case_Based': array([0., 0., 0., 0., 0., 1., 0.]),
#   'Rule_Learning': array([0., 0., 0., 0., 0., 0., 1.])}

# map() 会根据提供的函数对指定序列做映射。
# 这一句将string类型的label替换为onehot编码的label
labels_onehot = np.array(list(map(classes_dict.get, RawLabels)),
                             dtype=np.int32)
# print("onehot编码的论文类别（label）：\n",labels_onehot)
# [[0 0 0... 0 0 0]
#  [0 0 0... 1 0 0]
#  [0 1 0 ... 0 0 0]
#  ...
#  [0 0 0 ... 0 0 1]
#  [0 0 1 ... 0 0 0]
#  [0 0 0 ... 0 0 0]]
# print(labels_onehot.shape)
# (2708, 7)
labels = encode_onehot(idx_features_labels[:, -1])
'''
print(labels,labels.shape)
[[0 0 0 ... 0 0 1]
 [0 0 1 ... 0 0 0]
 [0 0 0 ... 1 0 0]
 ...
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 1 0]
 [0 0 0 ... 0 0 1]] (2708, 7)
'''

idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
idx_map = {j: i for i, j in enumerate(idx)}
edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
n = edges_unordered.flatten()
q = map(idx_map.get, edges_unordered.flatten())
m = list(map(idx_map.get, edges_unordered.flatten()))

edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
'''
print(idx)
print(idx.shape)

print(idx_map)
print(n)
print(q)

print(m)
print(edges)
'''

adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)


adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)   # coo变为csr
print(type(adj))
a = adj.todense()
b = np.array(a)
diff = compute_ppr(b, 0.2)


features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)

# print(features.shape)
features = normalize_features(features)
feat = np.array(features.todense())
# print(features)
adj = normalize_adj(adj + sp.eye(adj.shape[0]))
print(type(adj))
idx_train = range(140)
idx_val = range(200, 500)
idx_test = range(500, 1500)
# print(idx_train)


'''
print(idx)
print(idx_map)
print(edges_unordered)
print(edges)
'''
'''
print(labels)
print(labels.shape)
'''


features = torch.FloatTensor(np.array(features.todense()))

labels = torch.LongTensor(np.where(labels)[1])
print(np.unique(labels).shape[0])
'''
print(labels,labels.shape)
tensor([5, 0, 2,  ..., 3, 1, 5]) torch.Size([2708])
'''
adj = sparse_mx_to_torch_sparse_tensor(adj)
idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)
'''
print(idx_train)
print(idx_val)  
print(idx_test)
tensor([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,
         14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,
         28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,
         42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,
         56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,
         70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,
         84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,
         98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
        112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125,
        126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139])
'''

model = GCN(nfeat=features.shape[1],
            nhid=16,
            nclass=labels.max().item() + 1,
            dropout=0.05)
optimizer = optim.Adam(model.parameters(),
                       lr=0.01, weight_decay=5e-4)

a = 1
while a <2:
    t = time.time()
    model.train()
    optimizer.zero_grad()  #
    output = model(features, adj)
    print(output)
    print(output.shape)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])  # 准确率
    loss_train.backward()  # 反向传播
    optimizer.step()

    preds = output[idx_train].max(1)[1].type_as(labels[idx_train])
    correct = preds.eq(labels[idx_train]).double()
    correct = correct.sum()
    b=len(labels[idx_train])
    z = [idx_train]

    print(z)
    print(b)
    print(output[idx_train]) # 从output里的前140行训练集打印输出
    print(output[idx_train].shape)

    print(labels[idx_train]) # 从labels里的前140个标签类别打印输出
    print(labels[idx_train].shape)

    print(preds)
    print(preds.shape)
    print(correct)
    print(acc_train)

    a+=1



