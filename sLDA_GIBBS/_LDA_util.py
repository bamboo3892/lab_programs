# -*- coding: utf-8 -*-
import numpy as np
"""
Created on SEP 26 2018\n
This is cython source file, can be compiled to python module file\n
@author: takeda masaki
"""


def updateZ_LR(documents,
               alpha, beta,
               nWord_dk, nWord_kv, nWord_k,
               z,
               eta, rating):

    for d, document in enumerate(documents):
        for n, v in enumerate(document):
            z0 = z[d][n]
            nWord_k[z0] -= 1
            nWord_dk[d, z0] -= 1
            nWord_kv[z0, v] -= 1
            p0 = nWord_dk[d] + alpha
            p1 = nWord_kv[:, v] + beta[v]
            p2 = nWord_k + beta.sum()
            p3 = (np.eye(len(nWord_k)) + nWord_dk[d]) / len(document)  # K * K
            p3 = np.concatenate((np.ones((len(nWord_k), 1)), p3), 1).T  # K+1 * K
            p3 = np.dot(eta, p3)
            p3 = np.exp(rating[d] * p3) / (1 + np.exp(p3))
            p = p0 * p1 / p2 * p3
            p = p / p.sum()
            z0 = np.random.multinomial(1, p).argmax()
            z[d][n] = z0
            nWord_k[z0] += 1
            nWord_dk[d, z0] += 1
            nWord_kv[z0, v] += 1


def updateZ_BR(documents,
               alpha, beta,
               nWord_dk, nWord_kv, nWord_k,
               z,
               eta, dispersion, rating):

    for d, document in enumerate(documents):
        for n, v in enumerate(document):
            z0 = z[d][n]
            nWord_k[z0] -= 1
            nWord_dk[d, z0] -= 1
            nWord_kv[z0, v] -= 1
            p0 = nWord_dk[d] + alpha
            p1 = nWord_kv[:, v] + beta[v]
            p2 = nWord_k + beta.sum()
            p3 = (np.eye(len(nWord_k)) + nWord_dk[d]) / len(document)  # K * K
            p3 = np.concatenate((np.ones((len(nWord_k), 1)), p3), 1).T  # K+1 * K
            p3 = np.dot(eta, p3)
            p3 = 1.0 / np.sqrt(2 * np.pi * dispersion) * np.exp(-1.0 * np.square(rating[d] - p3) / 2.0 / dispersion)
            p = p0 * p1 / p2 * p3
            p = p / p.sum()
            z0 = np.random.multinomial(1, p).argmax()
            z[d][n] = z0
            nWord_k[z0] += 1
            nWord_dk[d, z0] += 1
            nWord_kv[z0, v] += 1
