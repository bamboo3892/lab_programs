# -*- coding: utf-8 -*-
"""
Created on OCT 14 2018

@author: takeda masaki
"""

import numpy as np
import json

from sLDA_GIBBS import sLDA


def testSLDA(pathMorphomes, pathTestMorphomes, pathResultFolder, *,
             textKey="morphomes", ratingKey="rating"):
    """ Example usage for sLDA class. """

    print("Start sLDA")

    def func0(path0, isClassifier):
        full = []
        docs = []
        rating = []
        with open(str(path0), "r", encoding="utf_8_sig") as file0:
            reviews = json.load(file0)
            for review in reviews:
                r = review[ratingKey]
                if(isClassifier):
                    if(r == 1 or r == 2):
                        rating.append(0)
                    elif(r == 5):
                        rating.append(1)
                    else:
                        continue
                else:
                    rating.append(r)
                full.append(review)
                docs.append(review[textKey])
        return full, docs, rating

    # read file
    full, docs, rating = func0(pathMorphomes, True)
    testFull, testDocs, testRating = func0(pathTestMorphomes, True)

    # make model, start learning
    model = sLDA.sLDA_LR(10, 0.01, 0.01, docs, rating, testDocs=testDocs, testRating=testRating)
    # model = sLDA.sLDA_BR(10, 0.01, 0.01, docs, rating)
    model.startLearning(100)

    # calc MSE
    error, prediction = model.calcMSE(docs, rating, suppressWarning=True)
    print("Mean square error(trainset): {:<.5f}".format(error))
    testError, testPrediction = model.calcMSE(testDocs, testRating, suppressWarning=True)
    print("Mean square error(testset): {:<.5f}".format(testError))

    if(isinstance(model, sLDA.sLDA_LR)):
        # calc accuracy
        accuracy, prediction = model.calcAccuracy(docs, rating)
        print("Accuracy(trainset): {:<.5f}".format(accuracy))
        testAccuracy, testPrediction = model.calcAccuracy(testDocs, testRating)
        print("Accuracy(testset): {:<.5f}".format(testAccuracy))

    # write result
    model.writeModelToFolder(pathResultFolder)

    for d, doc in enumerate(full):
        doc["theta_d"] = model.theta[d].tolist()
    with open(str(pathResultFolder.joinpath("documents.json")), "w", encoding="utf_8_sig") as file1:
        text = json.dumps(full, ensure_ascii=False)
        text = text.replace("},", "},\n")
        file1.write(text)

    print("Finish sLDA")
