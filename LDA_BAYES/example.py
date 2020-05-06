# -*- coding: utf-8 -*-
"""
Created on OCT 3 2018\n
LDA with Variational Bayesian estimation\n
@author: takeda masaki
"""

import csv
import json
import numpy as np
import time
import cProfile

from . import LDA
from . import LDA_util


def testLDA(pathMorphomes, pathTestMorphomes, pathResultFolder):
    """ Example usage for LDA class. """

    print("Start LDA")

    # read file
    docs = []
    with open(pathMorphomes, "r", encoding="utf8") as f0:
        docs = json.load(f0)

    # make model, and start learning
    model = LDA.LDA(5, 0.5, 0.0001, docs)
    # cProfile.run("model.startLearning(20)")
    model.startLearning(100)

    # write result
    pathResultFolder.mkdir(parents=True, exist_ok=True)
    with open(pathResultFolder.joinpath("theta.csv"), "w") as f1:
        writer = csv.writer(f1, lineterminator='\n')
        writer.writerows(model.theta)
    with open(pathResultFolder.joinpath("phi.csv"), "w") as f2:
        writer = csv.writer(f2, lineterminator='\n')
        writer.writerow(model.words)
        writer.writerows(model.phi)
        writer.writerow("")
        writer.writerow(["TOP 100 words in each topic"])
        top100 = []
        for k in range(model.K):
            top100_k = []
            words = model.words.copy()
            p0 = model.phi[k, :]
            for index in range(100):
                i0 = p0.argmax()
                top100_k.append(words[i0])
                p0 = np.delete(p0, i0)
                words = np.delete(words, i0)
            top100.append(top100_k)
        writer.writerows(top100)

    print("Finish LDA")
