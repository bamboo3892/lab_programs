# -*- coding: utf-8 -*-
"""
Created on OCT 3 2018\n
LDA with Variational Bayesian estimation\n
@author: takeda masaki
"""

import numpy as np
import sys
from scipy.special import digamma

from . import LDA_util


class LDA:

    def calcThetaAndPhi(self):
        self.theta = LDA_util.sumByLabel(self.q, self.idxToDoc, self.D)
        self.theta /= self.theta.sum(axis=1, keepdims=True)
        p0 = LDA_util.sumByLabel(self.q, self.idxToType, self.V)
        p1 = p0.sum(axis=0)
        self.phi = (p0 / p1).T
        # for d in range(self.D):
        #     td = self.q[self.idxToDoc == d, :].sum(axis=0)
        #     self.theta[d, :] = td / td.sum()
        # for v in range(self.V):
        #     pv = self.q[self.idxToType == v, :].sum(axis=0)
        #     self.phi[:, v] = pv / pv.sum()

    def __init__(self, K, alpha, beta, docs, dtype="f8"):
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
        """

        """ 0. init variables """
        # init learing materials
        self.documents = []
        # [
        #     [0, 1, 2, ..(word id)..],
        #     [document], [document], ...
        # ]
        self.words = []
        # ["word", "word", ...]
        for document in docs:
            for sentence in document:
                for word in sentence:
                    if(word not in self.words):
                        self.words.append(word)
        I = 0
        for document in docs:
            l0 = []
            for sentence in document:
                for word in sentence:
                    l0.append(self.words.index(word))
                I += len(sentence)
            self.documents.append(l0)
        D = len(self.documents)
        V = len(self.words)

        # init fields
        self.D = D  # number of documents
        self.V = V  # number of wordtypes
        self.K = K  # number of topics
        self.I = I  # number of words

        # init parameter of each probability
        self.ALPHA = alpha
        self.BETA = beta
        self.alpha = np.random.rand(D, K) + alpha  # parameter of theta
        self.beta = np.random.rand(K, V) + beta  # parameter of phi

        # init probability
        self.theta = np.zeros((D, K))  # word's topic probability for each document
        self.phi = np.zeros((K, V))  # word probability for each topic

        # init responsibility
        self.dtype = dtype
        self.q = np.zeros((I, K), dtype=dtype)
        self.idxToDoc = np.zeros(I, dtype="i4")  # convert consecutive index to document index
        self.idxToType = np.zeros(I, dtype="i4")  # convert consecutive index to wordtype
        self.idxToWord = np.zeros(I, dtype="i4")  # convert consecutive index to word index in each document
        idx = 0
        for nd, document in enumerate(self.documents):
            for nt, typ in enumerate(document):
                self.idxToDoc[idx] = nd
                self.idxToType[idx] = typ
                self.idxToWord[idx] = nt
                idx += 1
        a = self.alpha[self.idxToDoc, :]
        b = self.beta[:, self.idxToType].T
        self.q = np.exp(digamma(a) - digamma(a.sum(axis=1, keepdims=True)) + digamma(b) - digamma(b.sum(axis=0, keepdims=True)))
        self.q /= self.q.sum(axis=1, keepdims=True)

        self.calcThetaAndPhi()

    def startLearning(self, loop):
        print("Start leanring")
        print("Document: " + str(self.D))
        print("Word: " + str(self.I))
        print("Wordtypes: " + str(self.V))
        print("Topics: " + str(self.K))

        for lp in range(loop):
            sys.stdout.write("\r{:>4}/{:>4}".format(lp + 1, loop))
            sys.stdout.flush()

            a = self.alpha[self.idxToDoc, :]
            b = self.beta[:, self.idxToType].T
            self.q = np.exp(digamma(a) - digamma(a.sum(axis=1, keepdims=True)) + digamma(b) - digamma(b.sum(axis=0, keepdims=True)))
            self.q /= self.q.sum(axis=1, keepdims=True)
            self.alpha = self.ALPHA + LDA_util.sumByLabel(self.q, self.idxToDoc, self.D)
            self.beta = self.BETA + LDA_util.sumByLabel(self.q, self.idxToType, self.V).T
            # for d in range(self.D):
            #     self.alpha[d, :] = self.ALPHA + self.q[self.idxToDoc == d, :].sum(axis=0)
            # for v in range(self.V):
            #     self.beta[:, v] = self.BETA + self.q[self.idxToType == v, :].sum(axis=0)

        print("")
        self.calcThetaAndPhi()

    def startLearning_gpu(self, loop):
        print("")
