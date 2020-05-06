# -*- coding: utf-8 -*-
import numpy as np
cimport numpy as np
"""
Created on SEP 26 2018\n
This is cython source file, can be compiled to python module file\n
@author: takeda masaki
"""


def updateZ(documents,
            np.ndarray[np.double_t, ndim=1] alpha, np.ndarray[np.double_t, ndim=1] beta,
            np.ndarray[np.double_t, ndim=2] nWord_dk, np.ndarray[np.double_t, ndim=2] nWord_kv, np.ndarray[np.double_t, ndim=1] nWord_k,
            z):

    cdef int d, n, v, z0
    # cdef np.ndarray[DOUBLE_T, ndim=1] p0, p1, p2, p

    for d, document in enumerate(documents):
        for n, v in enumerate(document):
            z0 = z[d][n]
            nWord_k[z0] -= 1
            nWord_dk[d, z0] -= 1
            nWord_kv[z0, v] -= 1
            p0 = nWord_dk[d] + alpha
            p1 = nWord_kv[:, v] + beta[v]
            p2 = nWord_k + beta.sum()
            p = p0 * p1 / p2
            p = p / p.sum()
            z0 = np.random.multinomial(1, p).argmax()
            z[d][n] = z0
            nWord_k[z0] += 1
            nWord_dk[d, z0] += 1
            nWord_kv[z0, v] += 1

def updateZ_dk(documents,
            np.ndarray[np.double_t, ndim=1] alpha, np.ndarray[np.double_t, ndim=1] beta,
            np.ndarray[np.double_t, ndim=2] nWord_dk, np.ndarray[np.double_t, ndim=2] nWord_kv, np.ndarray[np.double_t, ndim=1] nWord_k,
            z):

    cdef int d, n, v, z0
    # cdef np.ndarray[DOUBLE_T, ndim=1] p0, p1, p2, p

    for d, document in enumerate(documents):
        for n, v in enumerate(document):
            z0 = z[d][n]
            nWord_dk[d, z0] -= 1
            p0 = nWord_dk[d] + alpha
            p1 = nWord_kv[:, v] + beta[v]
            p2 = nWord_k + beta.sum()
            p = p0 * p1 / p2
            p = p / p.sum()
            z0 = np.random.multinomial(1, p).argmax()
            z[d][n] = z0
            nWord_dk[d, z0] += 1
