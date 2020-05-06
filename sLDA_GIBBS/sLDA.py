# -*- coding: utf-8 -*-
"""
Created on OCT 14 2018\n
sLDA with Collapsed Gibbs sampling + fixed-point iteration + GeneralizedRegression\n
@author: takeda masaki
"""

import numpy as np
import random
from sklearn.linear_model import LogisticRegression, BayesianRidge
import csv

from LDA_GIBBS import LDA
from sLDA_GIBBS import LDA_util


class sLDA_base(LDA.LDA):

    def __init__(self, K, alpha, beta, docs, rating, *,
                 scoreCalcRate=10, testDocs=None, testRating=None):
        self.rating = rating
        self.testRating = testRating
        self.MSEHistory = []
        self.testMSEHistory = []
        self.initRegressionModel()
        super().__init__(K, alpha, beta, docs, arrangeInSentece=False,
                         scoreCalcRate=scoreCalcRate, testDocs=testDocs)

    def sampleInitZ(self):
        super().sampleInitZ()
        self.updateRegressionModel()

    def initRegressionModel(self):
        raise RuntimeError("This method must be overrided!!!")

    def updateZ(self):
        raise RuntimeError("This method must be overrided!!!")

    def updateRegressionModel(self):
        raise RuntimeError("This method must be overrided!!!")

    def calcExpectationFromModel(self, p_z):
        raise RuntimeError("This method must be overrided!!!")

    def learn(self):
        """
        Execute single learning loop\n
        Make sure to use #calcThetaAndPhi after learning loop
        """

        """ 1. sampleing z (Collapsed Gibbs sampling)"""
        self.updateZ()
        """ 2. estimate logistic regression's parameter"""
        self.updateRegressionModel()
        """ 3. update alpha and beta (fixed-point iteration) """
        self.updateAlphaAndBeta()

        self.countLearned += 1

    def addHistory(self):
        super().addHistory()
        self.MSEHistory.append(self.calcMSE(self.docs, self.rating)[0])
        if((self.testDocs is not None) and (self.testRating is not None)):
            self.testMSEHistory.append(self.calcMSE(self.testDocs, self.testRating)[0])

    def predict(self, documents, *,
                suppressWarning=False):
        """
        Returns the expectation of the document's rating\n
        If there is a document that does not have any trained word type,
        throws RuntimeError or returns negative number(-1) when suppressWarning is True
        """

        self.calcThetaAndPhi()

        prediction = []
        for document in documents:
            doc = []
            p_z = np.zeros(self.K)
            for sentence in document:
                for word in sentence:
                    if(word in self.words):
                        doc.append(word)
            if(len(doc) == 0):
                if(not suppressWarning):
                    raise RuntimeError("No valid word in param document")
                else:
                    prediction.append(-1)
                    continue
            for word in doc:
                p_z += self.phi[:, self.words.index(word)]
            p_z = p_z / p_z.sum()
            prediction.append(self.calcExpectationFromModel(p_z))

        return prediction

    def calcMSE(self, documents, rating, *,
                suppressWarning=True):
        """
        Calc MSE(mean square error) from param documents and corresponding rating\n
        When suppressWarning=False, if there is a document that does not have any trained word type,
        this method throws RuntimeError
        """
        prediction = self.predict(documents, suppressWarning)
        error = 0
        invalidD = 0
        for d in range(len(rating)):
            if(prediction[d] < 0):
                invalidD += 1
            else:
                error += (prediction[d] - rating[d]) ** 2
        error /= len(rating) - invalidD
        return error, prediction

    def writeModelToFolder(self, pathResultFolder):
        super().writeModelToFolder(pathResultFolder)
        if(self.scoreCalcRate > 0):
            with open(str(pathResultFolder.joinpath("phi.csv")), mode='a', encoding="utf_8_sig") as f:
                writer = csv.writer(f, lineterminator='\n')
                writer.writerow("")
                writer.writerow(["MSE history(trainingset)(recordedd every {} tick)".format(self.scoreCalcRate)])
                writer.writerow(self.MSEHistory)
                if(len(self.testMSEHistory) != 0):
                    writer.writerow(["MSE history(testset)(recordedd every {} tick)".format(self.scoreCalcRate)])
                    writer.writerow(self.testMSEHistory)


class sLDA_LR(sLDA_base):
    """
    sLDA with Logistic regression model\n
    \n
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
    scoreCalcRate: the rate score will be calclated from training data\n
                   Set negative to ban calclating score during learning\n
    """

    def __init__(self, K, alpha, beta, docs, rating, *,
                 scoreCalcRate=10, testDocs=None, testRating=None):
        self.accuracyHistory = []
        self.testAccuracyHistory = []
        super().__init__(K, alpha, beta, docs, rating,
                         scoreCalcRate=scoreCalcRate, testDocs=testDocs, testRating=testRating)

    def initRegressionModel(self):
        self.lr = LogisticRegression(solver="newton-cg")

    def updateZ(self):
        LDA_util.updateZ_LR(self.documents,
                            self.alpha, self.beta,
                            self.nWord_dk, self.nWord_kv, self.nWord_k,
                            self.z,
                            np.concatenate((self.lr.intercept_[0], self.lr.coef_[0, :]), None), self.rating)
        # LDA_util.updateZ_LR(self.documents,
        #                     self.alpha, self.beta,
        #                     self.nWord_dk, self.nWord_kv, self.nWord_k,
        #                     self.z,
        #                     np.concatenate((1, np.zeros(self.K)), None), self.rating)

    def updateRegressionModel(self):
        self.lr.fit((self.nWord_dk.T / self.nWord_d).T, self.rating)

    def addHistory(self):
        super().addHistory()
        self.accuracyHistory.append(self.calcAccuracy(self.docs, self.rating)[0])
        if((self.testDocs is not None) and (self.testRating is not None)):
            self.testAccuracyHistory.append(self.calcAccuracy(self.testDocs, self.testRating)[0])

    def calcExpectationFromModel(self, p_z):
        return self.lr.predict_proba(p_z.reshape(1, -1))[0, 1]

    def calcAccuracy(self, documents, rating, *,
                     suppressWarning=True):
        """
        Calc accuracy from param documents and corresponding rating\n
        When suppressWarning=False, if there is a document that does not have any trained word type,
        this method throws RuntimeError
        """
        prediction = self.predict(documents, suppressWarning)
        accuracy = 0
        invalidD = 0
        for d in range(len(rating)):
            if(prediction[d] < 0):
                invalidD += 1
            else:
                accuracy += prediction[d] if rating[d] == 1 else (1 - prediction[d])
        accuracy /= len(rating) - invalidD
        return accuracy, prediction

    def writeModelToFolder(self, pathResultFolder):
        super().writeModelToFolder(pathResultFolder)
        with open(str(pathResultFolder.joinpath("phi.csv")), mode='a', encoding="utf_8_sig") as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow("")
            writer.writerow(["accuracy history(trainingset)(recordedd every {} tick)".format(self.scoreCalcRate)])
            writer.writerow(self.accuracyHistory)
            if(len(self.testAccuracyHistory) != 0):
                writer.writerow(["accuracy history(testset)(recordedd every {} tick)".format(self.scoreCalcRate)])
                writer.writerow(self.testAccuracyHistory)
            writer.writerow("")
            writer.writerow(["eta"])
            writer.writerow([self.lr.intercept_[0]])
            writer.writerow(self.lr.coef_[0, :])


class sLDA_BR(sLDA_base):
    """
    sLDA with BayesianRidge regression model\n
    \n
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
    scoreCalcRate: the rate score will be calclated from training data\n
                   Set negative to ban calclating score during learning\n
    """

    def initRegressionModel(self):
        self.br = BayesianRidge(n_iter=100)

    def updateZ(self):
        LDA_util.updateZ_BR(self.documents,
                            self.alpha, self.beta,
                            self.nWord_dk, self.nWord_kv, self.nWord_k,
                            self.z,
                            np.concatenate((self.br.intercept_, self.br.coef_), None), 10, self.rating)

    def updateRegressionModel(self):
        self.br.fit((self.nWord_dk.T / self.nWord_d).T, self.rating)

    def calcExpectationFromModel(self, p_z):
        return self.br.predict(p_z.reshape(1, -1))[0]

    def writeModelToFolder(self, pathResultFolder):
        super().writeModelToFolder(pathResultFolder)
        with open(str(pathResultFolder.joinpath("phi.csv")), mode='a', encoding="utf_8_sig") as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow("")
            writer.writerow(["eta"])
            writer.writerow([self.br.intercept_])
            writer.writerow(self.br.coef_)
