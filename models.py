import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
from KCCA import KernelCCA
import numpy as np
from Kpca import kpca

'''
class Readout(nn.Module):
    def __init__(self):
        super(Readout, self).__init__()

    def forward(self, seq, msk):
        if msk is None:
            z = torch.unsqueeze(seq, 0)
            z = torch.mean(z, 1)
            return z
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.mean(seq * msk, 1) / torch.sum(msk)
'''


class Readout(nn.Module):
    def __init__(self):
        super(Readout, self).__init__()

    def forward(self, seq, msk):
        if msk is None:
            return torch.mean(seq, 1)
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.mean(seq * msk, 1) / torch.sum(msk)


'''

class Discriminator(nn.Module):
    def __init__(self, nhid):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(nhid, nhid, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c1, c2, h1, h2, h3, h4, s_bias1=None, s_bias2=None):

        c_x1 = c1.expand_as(h1).contiguous()
        c_x2 = c2.expand_as(h2).contiguous()

        # positive
        sc_1 = self.f_k(h2, c_x1)
        sc_1 = sc_1.permute(1, 0)

        sc_2 = self.f_k(h1, c_x2)
        sc_2 = sc_2.permute(1, 0)

        # negetive
        sc_3 = self.f_k(h4, c_x1)
        sc_3 = sc_3.permute(1, 0)

        sc_4 = self.f_k(h3, c_x2)
        sc_4 = sc_4.permute(1, 0)

        logits = torch.cat((sc_1, sc_2, sc_3, sc_4), 1)
        return logits
'''


class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c1, c2, h1, h2, h3, h4, s_bias1=None, s_bias2=None):
        c_x1 = torch.unsqueeze(c1, 1)
        c_x1 = c_x1.expand_as(h1).contiguous()
        c_x2 = torch.unsqueeze(c2, 1)
        c_x2 = c_x2.expand_as(h2).contiguous()
        # positive
        sc_1 = torch.squeeze(self.f_k(h2, c_x1), 2)
        sc_2 = torch.squeeze(self.f_k(h1, c_x2), 2)
        # negetive
        sc_3 = torch.squeeze(self.f_k(h4, c_x1), 2)
        sc_4 = torch.squeeze(self.f_k(h3, c_x2), 2)

        logits = torch.cat((sc_1, sc_2, sc_3, sc_4), 1)
        return logits


'''


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.gc3 = GraphConvolution(nfeat, nhid)
        self.gc4 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.read = Readout()
        self.disc = Discriminator(nhid)
        self.sigm = nn.Sigmoid()
        self.act = nn.PReLU()

    def forward(self, feature, adj, diff, shuf_fts, msk, samp_bias1, samp_bias2):

        h_1 = self.act(self.gc1(feature, adj))
        c_1 = self.read(h_1, msk)
        c_1 = self.act(c_1)
        h_11 = F.dropout(h_1, p=self.dropout, training=self.training)
        h_11 = self.gc2(h_11, adj)

        h_2 = self.act(self.gc3(feature, diff))
        c_2 = self.read(h_2, msk)
        c_2 = self.act(c_2)
        h_22 = F.dropout(h_2, p=self.dropout, training=self.training)
        h_22 = self.gc4(h_22, diff)

        h_3 = self.act(self.gc1(shuf_fts, adj))
        h_4 = self.act(self.gc3(shuf_fts, diff))

        score = self.disc(c_1, c_2, h_1, h_2, h_3, h_4, samp_bias1, samp_bias2)

        #kcca = KernelCCA(tau=1)
        #alpha, beta, lmbdas = kcca.learnModel(h_11, h_22)
        #p1, p2 = kcca.project(h_11, h_22, k=40)
        #p = np.c_[p1, p2]

        return F.log_softmax((h_11 + h_22), dim=1), score
        # return p, score

    def embed(self, x, adj, diff):
        h_1 = F.relu(self.gc1(x, adj))
        h_2 = F.relu(self.gc3(x, diff))
        return (h_1 + h_2).detach()

'''


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.gc3 = GraphConvolution(nfeat, nhid)
        self.gc4 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.read = Readout()
        self.disc = Discriminator(nhid)
        self.sigm = nn.Sigmoid()
        self.act = nn.PReLU()
        self.act2 = nn.ReLU()

    def forward(self, feature, adj, diff, shuf_fts, sparse, msk, samp_bias1, samp_bias2):
        h_1 = self.act(self.gc1(feature, adj, sparse))
        h_1 = torch.unsqueeze(h_1, 0)
        c_1 = self.read(h_1, msk)
        c_1 = self.sigm(c_1)
        h_11 = torch.squeeze(h_1)
        h_11 = F.dropout(h_11, p=self.dropout, training=self.training)
        h_11 = self.gc2(h_11, adj, sparse)

        h_2 = self.act(self.gc3(feature, diff, sparse))
        h_2 = torch.unsqueeze(h_2, 0)
        c_2 = self.read(h_2, msk)
        c_2 = self.sigm(c_2)
        h_22 = torch.squeeze(h_2)
        h_22 = F.dropout(h_22, p=self.dropout, training=self.training)
        h_22 = self.gc4(h_22, diff, sparse)

        h_3 = self.act(self.gc1(shuf_fts, adj, sparse))
        h_3 = torch.unsqueeze(h_3, 0)

        h_4 = self.act(self.gc3(shuf_fts, diff, sparse))
        h_4 = torch.unsqueeze(h_4, 0)
        score = self.disc(c_1, c_2, h_1, h_2, h_3, h_4, samp_bias1, samp_bias2)

        # Hkpca = (0.5*h_11+0.5*h_22)
        # Hkpca = Hkpca.cuda().data.cpu().numpy()
        # Hkpca = kpca(Hkpca, 7).cuda()

        return F.log_softmax((h_11 + h_22), dim=1), score
        # return F.log_softmax(Hkpca, dim=1), score

    def embed(self, x, adj, diff):
        h_1 = F.relu(self.gc1(x, adj))
        h_2 = F.relu(self.gc3(x, diff))
        return (h_1 + h_2).detach()


class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)
        self.sigm = nn.Sigmoid()
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = torch.log_softmax(self.fc(seq), dim=-1)
        return ret




