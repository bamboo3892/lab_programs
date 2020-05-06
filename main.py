# -*- coding: utf-8 -*-
"""
Created on SEP 19 2018

@author: takeda masaki
"""

from pathlib import Path
import gensim
import time

import formReviewToJson
import sepReviewsToMorphomes
import formDictInuiToJson
import LDA_GIBBS.example
import sLDA_GIBBS.example
import LLDA_GIBBS.example
import LDA_pyro.excute

import tensorDecomp.makeTensor
import tensorDecomp.matrixDecomp
import tensorDecomp.tensorDecomp


# file paths
# DATA_FOLDER = Path(__file__).absolute().parent.parent.joinpath("data")
DATA_FOLDER = Path("/mnt/nas/takeda/data")
ORIGINAL = DATA_FOLDER.joinpath("original")
FORMED = DATA_FOLDER.joinpath("formed")
MORPHOMES = DATA_FOLDER.joinpath("morphomes")
TENSOR = DATA_FOLDER.joinpath("tensor")
TAGGED = DATA_FOLDER.joinpath("tagged")
DICTIONARY = DATA_FOLDER.joinpath("dictionarys")
RESULT = DATA_FOLDER.joinpath("result")

if __name__ == '__main__':

    # flags: set true when you want to do it.
    b0 = False  # execute formHotelReview?
    b7 = False  # execute formGolfReview?
    b1 = False  # execute seperateReviewToMorphomes to hotel review?
    b8 = False  # execute seperateReviewToMorphomes to golf review?
    b4 = False  # excute LDA(gibbs) to hotel review?
    b9 = False  # excute LDA(gibbs) to golf review?
    b10 = False  # excute sLDA to golf review?
    b11 = False  # excute LLDA(gibbs) to golf review?
    b12 = False  # excute LLDA(gibbs) to golf review (multi process)?
    b13 = False  # makeTensor2
    b14 = False  # NMF
    b15 = False  # NTD
    b16 = False  # LDA LDA_pyro
    b17 = False  # sepSW LDA_pyro
    b18 = False  # sNTD LDA_pyro
    b19 = True  # eLLDA LDA_pyro

    if(b0):
        f0 = time.time()
        formReviewToJson.formHotelReview(ORIGINAL.joinpath("travel02_userReview00_20160304.txt"),
                                         FORMED.joinpath("travel_userReview.json"))
        print("(processed time: {:<.2f})".format(time.time() - f0))

    if(b7):
        f0 = time.time()
        formReviewToJson.formGolfReview(ORIGINAL.joinpath("golf01_userReview_20100713.txt"),
                                        FORMED.joinpath("golf_userReview.json"))
        print("(processed time: {:<.2f})".format(time.time() - f0))

    if(b1):
        f0 = time.time()
        sepReviewsToMorphomes.sepReviews(FORMED.joinpath("travel_userReview.json"),
                                         MORPHOMES.joinpath("travel_userReview1.json"),
                                         blacklist=r"(名詞-数)", whitelist=r"(^名詞)|(形容詞)",
                                         removeRateFromBtm=0.1, removeRateFromTop=0.15,
                                         nDocument=10000,
                                         minWordsInSentence=2)
        sepReviewsToMorphomes.sepReviews(FORMED.joinpath("travel_userReview.json"),
                                         MORPHOMES.joinpath("travel_userReview2.json"),
                                         blacklist=r"(名詞-数)", whitelist=r"(^名詞)|(形容詞)",
                                         removeRateFromBtm=0.1, removeRateFromTop=0.15,
                                         nDocument=10000, startIndex=10000,
                                         minWordsInSentence=2)
        print("(processed time: {:<.2f})".format(time.time() - f0))

    if(b8):
        f0 = time.time()
        sepReviewsToMorphomes.sepReviews(FORMED.joinpath("golf_userReview.json"),
                                         MORPHOMES.joinpath("golf_userReview1_10000.json"),
                                         blacklist=r"(名詞-数)", whitelist=r"(^名詞)|(形容詞)",
                                         blackWord=["こと", "よう", "ゴルフ場", "さ", "=", ",", "-", ":"],
                                         removeRateFromBtm=0.1, removeRateFromTop=0.15,
                                         nDocument=10000,
                                         minWordsInSentence=2)
        sepReviewsToMorphomes.sepReviews(FORMED.joinpath("golf_userReview.json"),
                                         MORPHOMES.joinpath("golf_userReview1_50000.json"),
                                         blacklist=r"(名詞-数)", whitelist=r"(^名詞)|(形容詞)",
                                         blackWord=["こと", "よう", "ゴルフ場", "さ", "=", ",", "-", ":"],
                                         removeRateFromBtm=0.1, removeRateFromTop=0.15,
                                         nDocument=50000,
                                         minWordsInSentence=2)
        sepReviewsToMorphomes.sepReviews(FORMED.joinpath("golf_userReview.json"),
                                         MORPHOMES.joinpath("golf_userReview2_10000.json"),
                                         blacklist=r"(名詞-数)", whitelist=r"(^名詞)|(形容詞)",
                                         blackWord=["こと", "よう", "ゴルフ場", "さ", "=", ",", "-", ":"],
                                         removeRateFromBtm=0.1, removeRateFromTop=0.15,
                                         nDocument=10000, startIndex=10000,
                                         minWordsInSentence=2)
        sepReviewsToMorphomes.sepReviews(FORMED.joinpath("golf_userReview.json"),
                                         MORPHOMES.joinpath("golf_userReview2_50000.json"),
                                         blacklist=r"(名詞-数)", whitelist=r"(^名詞)|(形容詞)",
                                         blackWord=["こと", "よう", "ゴルフ場", "さ", "=", ",", "-", ":"],
                                         removeRateFromBtm=0.1, removeRateFromTop=0.15,
                                         nDocument=50000, startIndex=50000,
                                         minWordsInSentence=2)
        print("(processed time: {:<.2f})".format(time.time() - f0))

    if(b4):
        f0 = time.time()
        LDA_GIBBS.example.testLDA(MORPHOMES.joinpath("travel_userReview1.json"),
                                  MORPHOMES.joinpath("travel_userReview2.json"),
                                  RESULT.joinpath("travel_userReview").joinpath("LDA_gibbs"))
        print("(processed time: {:<.2f})".format(time.time() - f0))

    if(b9):
        f0 = time.time()
        LDA_GIBBS.example.testLDA(MORPHOMES.joinpath("golf_userReview1_50000.json"),
                                  MORPHOMES.joinpath("golf_userReview2_50000.json"),
                                  RESULT.joinpath("golf_userReview").joinpath("LDA_gibbs"),
                                  w2vModel=gensim.models.Word2Vec.load(str(DICTIONARY.joinpath("ja.bin"))))
        print("(processed time: {:<.2f})".format(time.time() - f0))

    if(b10):
        f0 = time.time()
        sLDA_GIBBS.example.testSLDA(MORPHOMES.joinpath("golf_userReview1_50000.json"),
                                    MORPHOMES.joinpath("golf_userReview2_50000.json"),
                                    RESULT.joinpath("golf_userReview").joinpath("sLDA_gibbs"))
        print("(processed time: {:<.2f})".format(time.time() - f0))

    if(b11):
        f0 = time.time()
        LLDA_GIBBS.example.testLLDA(MORPHOMES.joinpath("golf_userReview1_50000.json"),
                                    MORPHOMES.joinpath("golf_userReview2_50000.json"),
                                    RESULT.joinpath("golf_userReview").joinpath("LLDA7_0"),
                                    w2vModel=gensim.models.Word2Vec.load(str(DICTIONARY.joinpath("ja.bin"))),
                                    fromPickle=False)
        print("(processed time: {:<.2f})".format(time.time() - f0))

    if(b12):
        f0 = time.time()
        LLDA_GIBBS.example.testLLDAs(MORPHOMES.joinpath("golf_userReview1_50000.json"),
                                     MORPHOMES.joinpath("golf_userReview2_50000.json"),
                                     #  RESULT.joinpath("golf_userReview", "2019_1_9", "LLDAS7_0"), "LLDAS7_",
                                     #  RESULT.joinpath("golf_userReview", "LLDAS15_5"), "LLDAS15_",
                                     #  RESULT.joinpath("golf_userReview", "LLDA8_0"), "LLDA8_",
                                     #  RESULT.joinpath("golf_userReview", "2019_1_9", "LLDAS8_0"), "LLDAS8_",
                                     #  RESULT.joinpath("golf-c", "LLDA6_10"), "LLDA6_",
                                     #  RESULT.joinpath("golf-c", "LLDAS7_5"), "LLDAS7_",
                                     #  RESULT.joinpath("golf_cf", "LLDA6_10"), "LLDA6_",
                                     RESULT.joinpath("golf_cf", "LLDA7_10"), "LLDA7_",
                                     w2vModel=gensim.models.Word2Vec.load(str(DICTIONARY.joinpath("ja.bin"))),
                                     fromPickle=False)
        print("(processed time: {:<.2f})".format(time.time() - f0))

    if(b13):
        f0 = time.time()
        tensorDecomp.makeTensor.makeTensor2(MORPHOMES.joinpath("golf_userReview1_50000.json"),
                                            TENSOR.joinpath("golf", "tensor2"),
                                            docType="golf")
        tensorDecomp.makeTensor.makeTensor3_golf(MORPHOMES.joinpath("golf_userReview1_50000.json"),
                                                 TENSOR.joinpath("golf", "tensor3"))
        print("(processed time: {:<.2f})".format(time.time() - f0))

    if(b14):
        f0 = time.time()
        tensorDecomp.matrixDecomp.doMatrixDecomp(TENSOR.joinpath("golf", "tensor2", "matrix.pickle"),
                                                 TENSOR.joinpath("golf", "tensor2", "words.dat"),
                                                 RESULT.joinpath("golf", "tensor", "tensor2", "NMF"),
                                                 7, method="nmf")
        print("(processed time: {:<.2f})".format(time.time() - f0))

    if(b15):
        f0 = time.time()
        tensorDecomp.tensorDecomp.doTensorDecomp_tensorly(TENSOR.joinpath("golf", "tensor3", "tensor.pickle"),
                                                          TENSOR.joinpath("golf", "tensor3", "words.dat"),
                                                          RESULT.joinpath("golf", "tensor", "tensor3", "TuckerDecomp_tensorly_nonnegative5",),
                                                          nComponent=(4, 10, 4),
                                                          nonnegative=True)
        print("(processed time: {:<.2f})".format(time.time() - f0))

    if(b16):
        f0 = time.time()
        LDA_pyro.excute.excuteLDA("LDA",
                                  TENSOR.joinpath("golf", "tensor2", "matrix.pickle"),
                                  TENSOR.joinpath("golf", "tensor2", "words.dat"),
                                  TENSOR.joinpath("golf", "tensor2", "documents.json"),
                                  RESULT.joinpath("golf", "LDA_pyro", "LDA", "tmp"),)
        print("(processed time: {:<.2f})".format(time.time() - f0))

    if(b17):
        f0 = time.time()
        LDA_pyro.excute.excuteLDA("sepSW",
                                  TENSOR.joinpath("golf", "tensor2", "matrix.pickle"),
                                  TENSOR.joinpath("golf", "tensor2", "words.dat"),
                                  TENSOR.joinpath("golf", "tensor2", "documents.json"),
                                  RESULT.joinpath("golf", "LDA_pyro", "sepSW", "tmp"),)
        print("(processed time: {:<.2f})".format(time.time() - f0))

    if(b18):
        f0 = time.time()
        LDA_pyro.excute.excuteLDA("sNTD",
                                  TENSOR.joinpath("golf", "tensor2", "matrix.pickle"),
                                  TENSOR.joinpath("golf", "tensor2", "words.dat"),
                                  TENSOR.joinpath("golf", "tensor2", "documents.json"),
                                  RESULT.joinpath("golf", "LDA_pyro", "sNTD", "tmp"),)
        print("(processed time: {:<.2f})".format(time.time() - f0))

    if(b19):
        f0 = time.time()
        LDA_pyro.excute.excuteLDA("eLLDA",
                                  TENSOR.joinpath("golf", "tensor2", "matrix.pickle"),
                                  TENSOR.joinpath("golf", "tensor2", "words.dat"),
                                  TENSOR.joinpath("golf", "tensor2", "documents.json"),
                                  RESULT.joinpath("golf", "LDA_pyro", "eLLDA", "tmp"),)
        print("(processed time: {:<.2f})".format(time.time() - f0))
