import numpy
import scipy.linalg
from numpy import dot, eye, ones, zeros
# from kernel_trick.kernel.AbstractKernel import AbstractKernel
import torch
"""
An implementation of the Kernel Canonincal Correlation Analysis (KCCA) algorithm. 
"""


class KernelCCA(object):
    def __init__(self, tau):

        self.tau = tau

    def learnModel(self, Kx, Ky):
        """
        Learn the KCCA  directions using set of examples given the numpy.ndarrays
        X and Y. If X and Y are matrices then their rows are examples, and they must
        have the same number of rows.

        :param X: The X examples.
        :type X: :class:`numpy.ndarray`

        :param Y: The Y examples.
        :type Y: :class:`numpy.ndarray`

        :returns alpha: The dual directions in the X space.
        :returns beta: The dual directions in the Y space.
        :returns lambda: The correlations for each projected dimension.
        """
        #Kx = numpy.array(Kx)
        #Ky = numpy.array(Ky)

        numExamples = Kx.shape[0]

        Kx = numpy.dot(Kx, Kx.T)
        Ky = numpy.dot(Ky, Ky.T)

        reg = 1e-5
        N = Kx.shape[0]
        I = eye(N)
        Z = numpy.zeros((N, N))

        R1 = numpy.c_[Kx, Ky]  # 两个4*4 变成4*8,按行拼
        R2 = R1
        R = 1. / 2 * numpy.r_[R1, R2]  # numpy.r_按列拼

        D1 = numpy.c_[Kx + reg * I, Z]
        D2 = numpy.c_[Z, Ky + reg * I]
        D = numpy.r_[D1, D2]

        (D, W) = scipy.linalg.eig(R, D)

        # Only select eigenvalues which are greater than zero
        W = W[:, D > 0]

        # We need to return those eigenvectors corresponding to positive eigenvalues
        self.alpha = W[0:numExamples, :]
        self.beta = W[numExamples:numExamples * 2, :]
        self.lmbdas = D[D > 0]

        alphaDiag = self.alpha.T.dot(Kx).dot(self.alpha)
        alphaDiag = alphaDiag + numpy.array(alphaDiag < 0, numpy.int)
        betaDiag = self.beta.T.dot(Ky).dot(self.beta)
        betaDiag = betaDiag + numpy.array(betaDiag < 0, numpy.int)
        self.alpha = numpy.dot(self.alpha, numpy.diag(1 / numpy.sqrt(numpy.diag(alphaDiag))))
        self.beta = numpy.dot(self.beta, numpy.diag(1 / numpy.sqrt(numpy.diag(betaDiag))))

        print(self.alpha,self.alpha.shape)

        return self.alpha, self.beta, self.lmbdas

    def project(self, testX, testY, k=None):
        """
        Project the examples in the KCCA subspace using set of test examples testX
        and testY. The number of projection directions is specified with k, and
        if this parameter is None then all directions are used.

        :param testX: The X examples to project.
        :type testX: :class:`numpy.ndarray`

        :param testY: The Y examples to project.
        :type testY: :class:`numpy.ndarray`

        :returns testXp: The projections of testX.
        :returns testYp: The projections of testY.
        """
        testX = numpy.array(testX)
        testY = numpy.array(testY)
        if k == None:
            k = self.alpha.shape[1]

        print(self.alpha.shape)
        print(self.alpha[:, 0:k].shape)
        print(abs(self.alpha[:, 0:k]).shape)

        return numpy.dot(testX, abs(self.alpha[:, 0:k])), numpy.dot(testY, abs(self.beta[:, 0:k]))

'''
h_11 = numpy.random.rand(4, 3)
h_22 = numpy.random.rand(4, 3)
kcca = KernelCCA(tau=1)
alpha, beta, lmbdas = kcca.learnModel(h_11, h_22)

p1, p2 = kcca.project(h_11, h_22, k=3)

print(p1.shape)
p = p1 + p2
print(h_11)
print(h_22)
print(p1)
print(p2)
print(p)
print(p.shape)
'''
