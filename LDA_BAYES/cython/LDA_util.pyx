# -*- coding: utf-8 -*-
import numpy as np
cimport numpy as np


def sumByLabel(a, label, nLabel):
    rtn = np.zeros((nLabel, a.shape[1]))

    for i in range(len(a)):
        rtn[label[i]] += a[i]

    return rtn