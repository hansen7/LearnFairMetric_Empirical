#  Copyright (c) 2020. Hanchen Wang, hw501@cam.ac.uk

import numpy as np
from itertools import compress
from numpy.matlib import repmat


class SERAPH(object):

    def __init__(self, X, Y, XT):
        self.X = X
        self.Y = Y
        self.XT = XT

    def construct_S_D(self):
        S = np.zeros((len(self.Y), len(self.Y)))
        D = np.zeros((len(self.Y), len(self.Y)))

        # Y = 0 means the label information is not available
        idx_ls = list(compress(range(len(self.Y)), self.Y != 0))

        for idx in idx_ls:
            for jdx in idx_ls:
                if idx != jdx:
                    if self.Y[idx] == self.Y[jdx]:
                        S[idx, jdx] = 1
                    else:
                        D[idx, jdx] = 1

        return S, D

    def computeObj_GP(self, eta, mu, lambda_, S, D, A):
        """
        Compute the Values of Objective Functions

        """
        epsp = 1e-12
        N, _ = self.X.shape
        XAX = np.dot(self.X @ A, self.X.T)
        X2 = np.diag(XAX)
        X2X = (np.zeros((len(X2), len(X2))) + X2)
        DD = X2X + X2X.T - 2 * XAX
        PP = 1 / (1 + np.exp(DD - eta)) + epsp
        ONE = np.ones((N, N))
        LL = np.zeros((N, N))
        LL[S == 1] = np.log(PP[S == 1])
        LL[D == 1] = np.log(ONE[D == 1] - PP[D == 1])
        U = ONE - S - D - np.diag(np.ones(N))

        if mu > 0:
            LL[U == 1] = mu * (PP[U == 1] * np.log(PP[U == 1]) +
                               (ONE[U == 1] - PP[U == 1]) * np.log(ONE[U == 1] - PP[U == 1]))

        objL = np.sum(LL) / 2 - lambda_ * np.trace(A)

        return objL

    def computeGrad_GP(self, eta, mu, lambda_, S, D, A):
        """
        Compute the gradient matrix of Objective Functions
        """
        epsp = 1e-12
        N, M = self.X.shape
        XAX = np.dot(self.X @ A, self.X.T)
        X2 = np.diag(XAX)
        X2X = (np.zeros((len(X2), len(X2))) + X2)
        DD = X2X + X2X.T - 2 * XAX
        PP = 1 / (1 + np.exp(DD - eta)) + epsp
        ONE = np.ones((N, N))

        CC = np.zeros((N, N))
        CC[S == 1] = PP[S == 1] - ONE[S == 1]
        CC[D == 1] = PP[D == 1]
        U = ONE - S - D - np.diag(np.ones(N))

        if mu > 0:
            CC[U == 1] = mu * (np.log(PP[U == 1]) - np.log(ONE[U == 1] - PP[U == 1])) * \
                         PP[U == 1] * (PP[U == 1] - ONE[U == 1])

        # print(np.sum(CC, 1).reshape(len(CC), 1).shape)
        # print((np.sum(CC, 1).reshape(len(CC), 1) * self.X).shape)

        grad = np.dot(self.X.T, np.sum(CC, 1).reshape(len(CC), 1) * self.X) - np.dot(self.X.T @ CC, self.X)
        grad = grad - lambda_ * np.diag(np.ones(M))
        return grad

    def solver_GP(self, eta, mu, lambda_, S, D, A_Init=None):
        """
        SERAPH Solver based on only GP
        :param eta: threshold to separate S and D
        :param mu: regularization parameter for entropy regularization,
                    0 means never use unlabeled data,
                    -1 means automatically compute mu=#labeled/#unlabeled
        :param lambda_: regularization parameter for trace-norm regularization
        :param S: set of similar pairs, S_ij=1 if (x_i,x_j) in S
        :param D: set of dissimilar pairs, D_ij=1 if (x_i,x_j) in D
        :param A_Init: initial distance metric, default is M/(M+1)*I
        :return: A_Final: final distance metric
                 iterGP: total number of iterations of GP

        """

        epsOut = 1e-3
        maxIter = 200
        maxTrial = 20

        N, M = self.X.shape
        if mu == -1:
            nLabeled = np.sum(S + D) / 2
            nUnlabeled = N * (N - 1) / 2 - nLabeled
            mu = nLabeled / nUnlabeled

        if not A_Init:
            A_Init = M / (M + 1) * np.diag(np.ones(M))

        iterGP = 0
        A_pre = (A_Init + A_Init.T) / 2
        objL_pre = self.computeObj_GP(eta, mu, lambda_, S, D, A_pre)
        A_best = A_pre
        objL_best = objL_pre
        normOut = 1e10

        while (normOut > epsOut) and (iterGP < maxIter):
            iterGP += 1

            # compute gradient
            grad = self.computeGrad_GP(eta, mu, lambda_, S, D, A_pre)

            # choose step size
            deltak = 0.1 * M / (np.linalg.norm(grad, ord='fro') * np.sqrt(iterGP))
            deltakk = deltak

            trial = 0
            while trial < maxTrial:
                trial += 1

                # gradient ascent
                A = A_pre + np.dot(deltak, grad)

                # projection
                A = (A + A.T) / 2
                b, V = np.linalg.eig(A)

                b = b * (b > 0)
                A = np.dot(V @ np.diag(b), V.T)

                # print(A)
                objL = self.computeObj_GP(eta, mu, lambda_, S, D, A)

                if objL >= objL_pre:  # success
                    objL_pre = objL
                    break
                else:  # fail
                    deltak = deltak / 2

            if trial == maxTrial:  # fail
                print('GP update fails...\n')
                A = A_pre + 2 * deltakk * grad
                A = (A + A.T) / 2
                b, V = np.linalg.eig(A)
                # b = np.diag(b)
                b = b * (b > 0)
                A = np.dot(V @ np.diag(b), V.T)

                objL = self.computeObj_GP(eta, mu, lambda_, S, D, A)

            elif objL >= objL_best:  # beat best
                A_best = A
                objL_best = objL

            normOut = np.linalg.norm(A_pre - A, ord='fro')
            A_pre = A
            objL_pre = objL

        A_Final = A_best

        return A_Final, iterGP

    def getProjection(self, A, d):
        A = (A + A.T) / 2
        D, V = np.linalg.eig(A)
        ds, di = np.sort(D)[::-1], np.argsort(D)[::-1]
        B = V[:, di[:d]]
        B = B * repmat(np.sign(np.sum(B, axis=0)), m=A.shape[0], n=1)
        P = B * repmat(np.real(np.sqrt(ds[:d])).T, m=A.shape[0], n=1)
        P = P.T

        return P

    def oneNN(self, x, A):
        # one-nearest-neighbor classifier
        X = self.X[(self.Y != 0).ravel(), :]
        Y = self.Y[(self.Y != 0).ravel()]
        D = np.diag(np.dot(X @ A, X.T)) - 2 * np.dot(X @ A, x.T)
        y = np.min(Y[D == min(D)])
        return y

    def fit(self):
        # SERACH, projection
        N = len(self.Y)
        S, D = self.construct_S_D()
        print('Finish Generating Similarity/Dissimilarity Constraints')

        nLabeled = np.sum(S + D) / 2
        nUnlabeled = N * (N - 1) / 2 - nLabeled
        mu = nLabeled / nUnlabeled

        A, _ = self.solver_GP(1, 100 * mu, 0, S, D)

        print(A)
        P = self.getProjection(A, 2)
        print(P)
        # X = np.dot(self.X, P.T)
        # XT = np.dot(self.XT, P.T)
        # X[:, 0], X[:, 1] = X[:, 1], X[:, 0]
        # XT[:, 0], XT[:, 1] = XT[:, 1], XT[:, 0]

        # SERAPH, classification
        N = len(self.XT)
        Y = self.Y
        YY = np.zeros((N, 1))

        for i in range(N):
            if Y[i] == 0:
                YY[i] = self.oneNN(self.XT[i, :], A)

        return YY


if __name__ == '__main__':
    from scipy.io import loadmat

    demo1 = loadmat('./SERAPH/demo1.mat')
    X = demo1['X']
    Y = demo1['Y']
    XT = demo1['XT']

    mytrial_ = SERAPH(X=X, Y=Y, XT=XT)

    S, D = mytrial_.construct_S_D()

    YY = mytrial_.fit()
    # print(YY)
