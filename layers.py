import math
import torch.nn as nn
import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

'''

class GraphConvolution(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(GraphConvolution, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU()

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    # Shape of seq: (batch, nodes, features)
    def forward(self, seq, adj):
        seq_fts = self.fc(seq)  
        # if sparse:
        out = torch.spmm(adj, torch.squeeze(seq_fts, 0))

        # else:
        # out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        return out


'''


class GraphConvolution(Module):

    # 初始化层：输入feature，输出feature，权重，偏移
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, sparse=False):
        support = torch.mm(input, self.weight)
        if sparse:
            output = torch.spmm(adj, support)
        else:
            output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
    # 打印形式是：GraphConvolution (输入特征 -> 输出特征)



