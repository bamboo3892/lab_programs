# -*- coding: utf-8 -*-
"""
Created on OCT 3 2018

@author: takeda masaki
"""

import numpy as np
import csv
import json

from LDA_GIBBS import LDA


def testLDA(pathMorphomes, pathTestMorphomes, pathResultFolder, *,
            textKey="morphomes", w2vModel=None):
    """ Example usage for LDA class. """

    print("Start LDA")

    # read file
    docs = []
    with open(str(pathMorphomes), "r", encoding="utf_8_sig") as f0:
        docs0 = json.load(f0)
        for doc in docs0:
            docs.append(doc[textKey])
    testDocs = []
    with open(str(pathMorphomes), "r", encoding="utf_8_sig") as f1:
        docs1 = json.load(f1)
        for doc in docs1:
            testDocs.append(doc[textKey])

    # make model, and start learning
    model = LDA.LDA(7, 0.01, 0.01, docs, testDocs=testDocs, scoreCalcRate=10, w2vModel=w2vModel)
    model.startLearning(1)

    # calc perplexity
    # print("Perplexity(trainset): {}".format(model.calcPerplexity2(docs, 0.2)))
    # print("Perplexity(testset): {}".format(model.calcPerplexity2(testDocs, 0.2)))

    # write result
    model.writeModelToFolder(pathResultFolder)
    # make documents.json for analysis
    for d, doc in enumerate(docs0):
        doc["theta_d"] = model.theta[d].tolist()
        doc["nWord_dk"] = model.nWord_dk[d].tolist()
        doc["theta2_d"] = model.theta2[d].tolist()
        doc["theta3_d"] = model.theta3[d].tolist()
    with open(str(pathResultFolder.joinpath("documents.json")), "w", encoding="utf_8_sig") as file1:
        text = json.dumps(docs0, ensure_ascii=False)
        text = text.replace("},", "},\n")
        file1.write(text)

    print("Finish LDA")
