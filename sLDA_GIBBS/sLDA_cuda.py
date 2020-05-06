# -*- coding: utf-8 -*-
"""
Created on OCT 14 2018\n
sLDA with Collapsed Gibbs sampling + fixed-point iteration + Logistic regression\n
@author: takeda masaki
"""

import numba.cuda as cuda
import numpy as np
import random
from sklearn.linear_model import LogisticRegression

from LDA_GIBBS import LDA
from sLDA_GIBBS import LDA_util


class sLDA_cuda(sLDA.sLDA):
    """
    K         : number of topics\n
    alpha     : initial parameter of theta\n
                same value between documents\n
    beta      : initial parameter of phi\n
                same value between topics\n
    docs      : documents\n
                [
                    [
                        [
                            [word], [word], ...
                        ], [sentence], [sentence], ...
                    ], [document], [document], ...
                ]
    rating    : rating label list\n
                Should be 0 or 1(?)\n
    """

    def __init__(self, K, alpha, beta, docs, rating):
        super().__init__(K, alpha, beta, docs, rating)

# @cuda.jit('void(int32[:,:],float32[:],float32[:],int32[:,:],int32[:,:],int32[:,:],int32[:,:],float32[:],int32[:])', device=True)


@cuda.jit
def updateZ_cuda(documents, indexStartDocument,
                 nWord_d, nWord_dk, nWord_kv, nWord_k,
                 alpha, beta,
                 eta, rating,
                 z):
    """
    DO NOT USE THIS METHOD
    """
    d = cuda.grid(1)
    D = len(nWord_d)
    K = alpha.shape()
    V = beta.shape()
    for n in range(nWord_d[d]):
        index = indexStartDocument[d] + n
        z = z2[index]
        for k in range(K):
            # TODO
            aaa = 0

    def startLearning(self, loop):
        print("Start leanring")
        print("Document: " + str(self.D))
        print("Wordtypes: " + str(self.V))
        print("Topics: " + str(self.K))

        document2 = []
        z2 = []
        indexStartDocument = [0]
        for d in range(self.D):
            document2.extend(self.document[d])
            z2.extend(self.z[d])
            indexStartDocument.append(self.nWord_d[d])

        for i in range(loop):
            sys.stdout.write("\r{:>4}/{:>4}".format(i + 1, loop))
            sys.stdout.flush()
            updateZ_cuda[3, 3](document2, indexStartDocument,
                               self.nWord_d, self.nWord_dk, self.nWord_kv, self.nWord_k,
                               self.alpha, self.beta,
                               np.concatenate((self.lr.intercept_[0], self.lr.coef_[0, :]), None), self.rating,
                               z2)
            self.fitLogisticModel()
            self.updateAlphaAndBeta()
        print("")

        index = 0
        for d in range(self.D):
            for n in range(self.nWord_d[d]):
                z[d][n] = z2[index]
                index += 1

        self.calcThetaAndPhi()
