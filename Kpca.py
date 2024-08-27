from scipy.spatial.distance import pdist,squareform
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch
from numpy import *


class KPCA():
    def __init__(self, kernel='rbf'):
        self.kernel = kernel

    def fit_transform_plot(self, X, y):
        if self.kernel == 'None':
            C = np.cov(X.T)
            eigvals, eigvecs = np.linalg.eigh(C)
            arg_max = eigvals.argsort()[-2:]
            eigvecs_max = eigvecs[:, arg_max]
            K = X
        else:
            if self.kernel == 'linear':
                K = np.dot(X, X.T)
            elif self.kernel == 'log':
                dists = pdist(X) ** 0.2
                mat = squareform(dists)
                K = -np.log(1 + mat)
            elif self.kernel == 'rbf':
                dists = pdist(X) ** 2
                mat = squareform(dists)
                beta = 10
                K = np.exp(-beta * mat)
            else:
                print('kernel error!')
                return None
            N = K.shape[0]
            one_n = np.ones([N, N]) / N
            K_hat = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
            eigvals, eigvecs = np.linalg.eigh(K_hat)
            arg_max = eigvals.argsort()[-2:]
            eigvecs_max = eigvecs[:,arg_max]
        X_new = np.dot(K, eigvecs_max)
        for i in range(2):
            tmp = y == i
            Xi = X_new[tmp]
            plt.scatter(Xi[:,0], Xi[:,1])
        plt.show()


def kpca(k, topN):
    K = np.dot(k, k.T)
    '''
    dists = pdist(k) ** 0.2
    mat = squareform(dists)
    K = -np.log(1 + mat)
    '''

    '''
    dists = pdist(k) ** 2
    mat = squareform(dists)
    beta = 10
    K = np.exp(-beta * mat)
    '''
    N = K.shape[0]
    one_n = np.ones([N, N]) / N
    K_hat = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
    eigvals, eigvecs = np.linalg.eigh(K_hat)
    arg_max = eigvals.argsort()[-1: -(topN + 1): -1]
    eigvecs_max = eigvecs[:, arg_max]
    for i in range(topN):
        eigvecs_max[:, i] = eigvecs_max[:, i] / sqrt(eigvals[arg_max[i]])
    X_new = np.dot(K_hat, eigvecs_max)
    X_new = torch.FloatTensor(X_new)
    return  X_new

'''
k = np.random.rand(2708,7)
k = torch.FloatTensor(k)
k = F.relu(k)
print(k)
k = np.array(k)
kk = kpca(k,7)
print(kk,kk.shape)
kk = F.log_softmax(kk,dim=1)
print(kk)
'''


'''
if __name__ == '__main__':
    X, y = make_circles(n_samples=400, factor=.3, noise=.05, random_state=0)
    print('aa',X.shape,y.shape)
    print(X)
    print(y)
    kpca = KPCA('linear')
    kpca.fit_transform_plot(X, y)
'''