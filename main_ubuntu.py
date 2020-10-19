# -*- coding: utf-8 -*-
"""
Created on OCT 26 2018

@author: takeda masaki
"""

from pathlib import Path
import time
import json
import gensim
import shutil

import formReviewToJson
import sepReviewsToMorphomes

import LDA_GIBBS.example
import sLDA_GIBBS.example
import LLDA_GIBBS.example
import LDA_pyro.excute
import LDA_pytorch.excute
import analysis.analyzeResult

import tensorDecomp.makeTensor
import tensorDecomp.matrixDecomp
import tensorDecomp.tensorDecomp

import analysis.cca
import analysis.analyzeHealthCheck

# file paths
DATA_FOLDER = Path("/mnt/nas/prj_qol/福間研共同研究/特定保健指導/data")
ORIGINAL = DATA_FOLDER.joinpath("original")
HEALTH_CHECK = ORIGINAL.joinpath("Health_check_up_2329.csv")
HEALTH_CHECK_REFINED = DATA_FOLDER.joinpath("analysis_healthcheck", "hc_refined.pickle")
FORMED = DATA_FOLDER.joinpath("formed")
MORPHOMES = DATA_FOLDER.joinpath("morphomes")
TENSOR = DATA_FOLDER.joinpath("tensor")
RESULT = DATA_FOLDER.joinpath("result")
ANALYSIS_HEALTH_CHECK = DATA_FOLDER.joinpath("analysis_healthcheck")

DICTIONARY = Path(__file__).absolute().parent.parent.joinpath("data").joinpath("dictionarys")

HC_MEASUREMENT_KEYS = analysis.analyzeHealthCheck.dataLabels
HC_HABIT_KEYS = analysis.analyzeHealthCheck.habitLabels
HC_MEDICINE_KEYS = analysis.analyzeHealthCheck.medicineLabels
HC_HABIT_LEVELS = analysis.analyzeHealthCheck.habitLevels2
HC_MEDICINE_LEVELS = analysis.analyzeHealthCheck.medicineLevels

MC_TEXT_KEYS_FOR_TYPE = ["p_r_explan_iyoku", "p_r_exer", "p_r_other_text",
                         "p_r_drink_text", "p_r_snack_text", "p_r_sake_text", "p_r_sleep_text", "p_r_other"]
MC_TEXT_KEYS_FOR_DIRECTION = ["p_r_tgtset_explan"]
MULTI_CHANNEL_KEYS = MC_TEXT_KEYS_FOR_TYPE + MC_TEXT_KEYS_FOR_DIRECTION


if __name__ == '__main__':

    b21 = False  # save health check file as pandas

    b0 = False  # execute formReviewToJson?
    b1 = False  # execute sepReviewsToMorphomes?
    b2 = False  # excute LDA(gibbs)?
    b20 = False  # excute LDA_pyro?
    b24 = False  # excute LDA_pytorch?
    b3 = False  # excute sLDA?
    b4 = False  # excute LLDA
    b5 = False  # excute analyze

    b6 = False  # makeTensor2
    b7 = False  # makeTensor3
    b8 = False  # PCA
    b14 = False  # NMF
    b9 = False  # CP decomp torch
    b10 = False  # CP decomp sktensor
    b12 = False  # Tucker decomp sktensor
    b13 = False  # Tucker decomp tensorly
    b15 = False  # Tucker decomp tensorly (NTD)

    b16 = False  # execute sepReviewsToMorphomes for multi channel?
    b17 = False  # makeTensor2 for multi channel
    b18 = False  # excute LDA to individual channel
    b19 = False  # excute MCLDA
    b22 = False  # excute MCLDAnum
    b23 = False  # excute MCLDAnum_only
    b25 = False  # excute MCLDA for type(torch)
    b27 = False  # excute MCLDA for direction(torch)
    b26 = False  # excute MCLDA for each K (torch)
    b28 = True  # excute CCA

    b11 = False  # HC statistics


    if(b21):
        f0 = time.time()
        analysis.analyzeHealthCheck.refineData(HEALTH_CHECK, HEALTH_CHECK_REFINED)
        print("(processed time: {:<.2f})".format(time.time() - f0))

    if(b0):
        f0 = time.time()
        # formReviewToJson.formSOMPO(ORIGINAL.joinpath("Original_data_SOMPO_20180927.csv"),
        #                            ORIGINAL.joinpath("furikaeri_k_id_50117.csv"),
        #                            ORIGINAL.joinpath("新プログラムで登録した人リスト.csv"),
        #                            FORMED.joinpath("sompo.json"))
        # formReviewToJson.formSOMPO(ORIGINAL.joinpath("Original_data_SOMPO_20180927.csv"),
        #                            ORIGINAL.joinpath("furikaeri_k_id_50117.csv"),
        #                            ORIGINAL.joinpath("新プログラムで登録した人リスト.csv"),
        #                            FORMED.joinpath("sompo_message.json"),
        #                            textLabel="ano_p_r_message")
        formReviewToJson.formSOMPO(ORIGINAL.joinpath("Original_data_SOMPO_20180927.csv"),
                                   ORIGINAL.joinpath("furikaeri_k_id_50117.csv"),
                                   ORIGINAL.joinpath("新プログラムで登録した人リスト.csv"),
                                   FORMED.joinpath("sompo_first.json"),
                                   firstGuidanceOnly=True)
        print("(processed time: {:<.2f})".format(time.time() - f0))

    if(b1):
        f0 = time.time()
        # sepReviewsToMorphomes.sepReviews(FORMED.joinpath("sompo.json"),
        #                                  MORPHOMES.joinpath("sompo1_10000.json"),
        #                                  TextKey="p_r_tgtset_explan",
        #                                  blacklist=r"(名詞-数)", whitelist=r"(^名詞)",
        #                                  blackWord=["量", "称賛", "ため", "為", "改善", "目標", "説明", "確認"],
        #                                  removeRateFromBtm=0.05, removeRateFromTop=0.15,
        #                                  nDocument=10000,
        #                                  minWordsInSentence=2)
        # sepReviewsToMorphomes.sepReviews(FORMED.joinpath("sompo.json"),
        #                                  MORPHOMES.joinpath("sompo2_10000.json"),
        #                                  textKey="p_r_tgtset_explan",
        #                                  blacklist=r"(名詞-数)", whitelist=r"(^名詞)",
        #                                  blackWord=["量", "称賛", "ため", "為", "改善", "目標", "説明", "確認"],
        #                                  removeRateFromBtm=0.05, removeRateFromTop=0.15,
        #                                  nDocument=10000, startIndex=10000,
        #                                  minWordsInSentence=2)
        # sepReviewsToMorphomes.sepReviews(FORMED.joinpath("sompo.json"),
        #                                  MORPHOMES.joinpath("sompo1_30000.json"),
        #                                  textKey="p_r_tgtset_explan",
        #                                  blacklist=r"(名詞-数)", whitelist=r"(^名詞)",
        #                                  blackWord=["量", "称賛", "ため", "為", "改善", "目標", "説明", "確認"],
        #                                  removeRateFromBtm=0.05, removeRateFromTop=0.15,
        #                                  nDocument=30000,
        #                                  minWordsInSentence=2)
        # sepReviewsToMorphomes.sepReviews(FORMED.joinpath("sompo.json"),
        #                                  MORPHOMES.joinpath("sompo2_30000.json"),
        #                                  textKey="p_r_tgtset_explan",
        #                                  blacklist=r"(名詞-数)", whitelist=r"(^名詞)",
        #                                  blackWord=["量", "称賛", "ため", "為", "改善", "目標", "説明", "確認"],
        #                                  removeRateFromBtm=0.05, removeRateFromTop=0.15,
        #                                  nDocument=30000, startIndex=30000,
        #                                  minWordsInSentence=2)
        # sepReviewsToMorphomes.sepReviews(FORMED.joinpath("sompo.json"),
        #                                  MORPHOMES.joinpath("sompo_full.json"),
        #                                  textKey="p_r_tgtset_explan",
        #                                  blacklist=r"(名詞-数)", whitelist=r"(^名詞)",
        #                                  blackWord=["量", "称賛", "ため", "為", "改善", "目標", "説明", "確認"],
        #                                  removeRateFromBtm=0.03, removeRateFromTop=0.15,
        #                                  minWordsInSentence=2)

        # first only (テストセットはトレーニングセットと上から禁止する語彙を同じにした)
        sepReviewsToMorphomes.sepReviews(FORMED.joinpath("sompo_first.json"),
                                         MORPHOMES.joinpath("sompo_first_10000_1.json"),
                                         inputTextKey="p_r_tgtset_explan",
                                         outputTextKey="p_r_tgtset_explan_morphomes",
                                         blacklist=r"(名詞-数)", whitelist=r"(^名詞)",
                                         blackWord=["量", "称賛", "ため", "為", "改善", "目標", "説明", "確認"],
                                         removeRateFromBtm=0.03, removeRateFromTop=0.15,
                                         nDocument=10000, startIndex=0,
                                         minWordsInSentence=2)
        sepReviewsToMorphomes.sepReviews(FORMED.joinpath("sompo_first.json"),
                                         MORPHOMES.joinpath("sompo_first_10000_2.json"),
                                         inputTextKey="p_r_tgtset_explan",
                                         outputTextKey="p_r_tgtset_explan_morphomes",
                                         blacklist=r"(名詞-数)", whitelist=r"(^名詞)",
                                         blackWord=["こと", "説明", "体重", "量", "事", "よう",
                                                    "量", "称賛", "ため", "為", "改善", "目標", "説明", "確認"],
                                         removeRateFromBtm=0.03, removeRateFromTop=0.0,
                                         nDocument=10000, startIndex=10000,
                                         minWordsInSentence=2)

        # message
        # sepReviewsToMorphomes.sepReviews(FORMED.joinpath("sompo_message.json"),
        #                                  MORPHOMES.joinpath("sompo_message.json"),
        #                                  textKey="ano_p_r_message",
        #                                  #  blacklist=r"(名詞-数)", whitelist=r"(^名詞)",
        #                                  blacklist=r"(名詞-数)", whitelist=r"(^名詞)|(^形容詞)|(^動詞)",
        #                                  blackWord=["こと", "事", "ため", "為", "No", "text", "-", "*", "(", ")", "〜", "/",
        #                                             "れる", "なる", "いる", "こと", "する", "よう", "ある",
        #                                             "おる", "思う", "みる", "日", "できる", "時", "中"],
        #                                  removeRateFromBtm=0.03, removeRateFromTop=0.0,
        #                                  minWordsInSentence=2)

        print("(processed time: {:<.2f})".format(time.time() - f0))

    if(b2):
        f0 = time.time()
        LDA_GIBBS.example.testLDA(MORPHOMES.joinpath("sompo1_30000.json"),
                                  MORPHOMES.joinpath("sompo2_30000.json"),
                                  RESULT.joinpath("p_r_tgtset_explan", "sompo", "LDA_gibbs7"),
                                  w2vModel=gensim.models.Word2Vec.load(str(DICTIONARY.joinpath("ja.bin"))))
        print("(processed time: {:<.2f})".format(time.time() - f0))

    if(b20):
        f0 = time.time()
        LDA_pyro.excute.excuteLDAFromPath("LDA",
                                          TENSOR.joinpath("tensor2", "matrix.pickle"),
                                          TENSOR.joinpath("tensor2", "words.dat"),
                                          MORPHOMES.joinpath("sompo1_10000.json"),
                                          RESULT.joinpath("p_r_tgtset_explan", "pyro", "LDA", "tmp"))
        # LDA_pyro.excute.excuteLDAFromPath("LDA_original",
        #                                   TENSOR.joinpath("tensor2", "matrix.pickle"),
        #                                   TENSOR.joinpath("tensor2", "words.dat"),
        #                                   MORPHOMES.joinpath("sompo1_10000.json"),
        #                                   RESULT.joinpath("p_r_tgtset_explan", "pyro", "LDA_original"))
        # LDA_pyro.excute.excuteLDAFromPath("LDA_mcmc",
        #                                   TENSOR.joinpath("tensor2", "matrix.pickle"),
        #                                   TENSOR.joinpath("tensor2", "words.dat"),
        #                                   MORPHOMES.joinpath("sompo1_10000.json"),
        #                                   RESULT.joinpath("p_r_tgtset_explan", "pyro", "LDA_mcmc"))
        print("(processed time: {:<.2f})".format(time.time() - f0))

    if(b24):
        f0 = time.time()
        LDA_pytorch.excute.excuteLDA(MORPHOMES.joinpath("sompo1_30000.json"),
                                     RESULT.joinpath("p_r_tgtset_explan", "pytorch", "LDA", "tmp"))
        print("(processed time: {:<.2f})".format(time.time() - f0))

    if(b3):
        f0 = time.time()
        sLDA_GIBBS.example.testSLDA(MORPHOMES.joinpath("sompo1_30000.json"),
                                    MORPHOMES.joinpath("sompo2_30000.json"),
                                    RESULT.joinpath("p_r_tgtset_explan", "sompo", "sLDA_gibbs"),
                                    ratingKey=None)
        print("(processed time: {:<.2f})".format(time.time() - f0))

    if(b4):
        f0 = time.time()
        # resultLLDA = RESULT.joinpath("p_r_tgtset_explan", "sompo_18", "LLDAS30", "LLDAS30_0")
        # LLDA_GIBBS.example.testLLDA(MORPHOMES.joinpath("sompo_full003.json"),
        #                             MORPHOMES.joinpath("sompo1_30000.json"),
        #                             resultLLDA,
        #                             w2vModel=gensim.models.Word2Vec.load(str(DICTIONARY.joinpath("ja.bin"))),
        #                             fromPickle=False)
        # resultLLDA = RESULT.joinpath("ano_p_r_message", "sompo_message", "LLDAS30_100", "LLDAS30_1")
        # LLDA_GIBBS.example.testLLDA(MORPHOMES.joinpath("sompo_message.json"),
        #                             MORPHOMES.joinpath("sompo_message_1000.json"),
        #                             resultLLDA,
        #                             w2vModel=gensim.models.Word2Vec.load(str(DICTIONARY.joinpath("ja.bin"))),
        #                             fromPickle=False)
        resultLLDA = RESULT.joinpath("p_r_tgtset_explan", "sompo_19_first", "LLDAS30", "tmp")
        LLDA_GIBBS.example.testLLDA(MORPHOMES.joinpath("sompo_first_10000_1.json"),
                                    MORPHOMES.joinpath("sompo_first_10000_2.json"),
                                    resultLLDA,
                                    w2vModel=gensim.models.Word2Vec.load(str(DICTIONARY.joinpath("ja.bin"))),
                                    fromPickle=True)
        print("(processed time: {:<.2f})".format(time.time() - f0))

    if(b5):
        f0 = time.time()
        analysis.analyzeResult.analyze(resultLLDA, resultLLDA, RESULT.joinpath("template.xlsm"),
                                       HEALTH_CHECK,
                                       #    ANALYSIS_HEALTH_CHECK.joinpath("anova_p.csv")
                                       )
        print("(processed time: {:<.2f})".format(time.time() - f0))


    # --------------------------------------------------------- TENSOR ---------------------------------------------------------

    if(b6):
        f0 = time.time()
        tensorDecomp.makeTensor.makeTensor2(MORPHOMES.joinpath("sompo_full003.json"),
                                            MORPHOMES.joinpath("wordcounts.csv"),
                                            TENSOR.joinpath("tensor2"))
        print("(processed time: {:<.2f})".format(time.time() - f0))

    if(b7):
        f0 = time.time()
        tensorDecomp.makeTensor.makeTensor3(MORPHOMES.joinpath("sompo_full003.json"),
                                            MORPHOMES.joinpath("wordcounts.csv"),
                                            TENSOR.joinpath("tensor3"))
        print("(processed time: {:<.2f})".format(time.time() - f0))

    if(b8):
        f0 = time.time()
        tensorDecomp.matrixDecomp.doMatrixDecomp(TENSOR.joinpath("tensor2", "matrix.pickle"),
                                                 TENSOR.joinpath("tensor2", "words.dat"),
                                                 RESULT.joinpath("p_r_tgtset_explan", "tensor", "tensor2", "PCA"))
        print("(processed time: {:<.2f})".format(time.time() - f0))

    if(b14):
        f0 = time.time()
        tensorDecomp.matrixDecomp.doMatrixDecomp(TENSOR.joinpath("tensor2", "matrix.pickle"),
                                                 TENSOR.joinpath("tensor2", "words.dat"),
                                                 RESULT.joinpath("p_r_tgtset_explan", "tensor", "tensor2", "NMF"), method="nmf")
        print("(processed time: {:<.2f})".format(time.time() - f0))

    if(b9):
        f0 = time.time()
        tensorDecomp.tensorDecomp.doCPDecomp_torch(TENSOR.joinpath("tensor3", "tensor.pickle"),
                                                   TENSOR.joinpath("tensor3", "words.dat"),
                                                   TENSOR.joinpath("tensor3", "mIDs.dat"),
                                                   RESULT.joinpath("p_r_tgtset_explan", "tensor", "tensor3", "CPDecomp"))
        print("(processed time: {:<.2f})".format(time.time() - f0))

    if(b10):
        f0 = time.time()
        tensorDecomp.tensorDecomp.doTensorDecomp_sktensor(TENSOR.joinpath("tensor3", "tensor.pickle"),
                                                          TENSOR.joinpath("tensor3", "words.dat"),
                                                          TENSOR.joinpath("tensor3", "mIDs.dat"),
                                                          RESULT.joinpath("p_r_tgtset_explan", "tensor", "tensor3", "CPDecomp_sktensor"), "cp_als")
        print("(processed time: {:<.2f})".format(time.time() - f0))

    if(b12):
        f0 = time.time()
        tensorDecomp.tensorDecomp.doTensorDecomp_sktensor(TENSOR.joinpath("tensor3", "tensor.pickle"),
                                                          TENSOR.joinpath("tensor3", "words.dat"),
                                                          TENSOR.joinpath("tensor3", "mIDs.dat"),
                                                          RESULT.joinpath("p_r_tgtset_explan", "tensor", "tensor3", "TuckerDecomp_sktensor"),
                                                          "hooi")
        print("(processed time: {:<.2f})".format(time.time() - f0))

    if(b13):
        f0 = time.time()
        tensorDecomp.tensorDecomp.doTensorDecomp_tensorly(TENSOR.joinpath("tensor3", "tensor.pickle"),
                                                          TENSOR.joinpath("tensor3", "words.dat"),
                                                          TENSOR.joinpath("tensor3", "mIDs.dat"),
                                                          RESULT.joinpath("p_r_tgtset_explan", "tensor", "tensor3", "TuckerDecomp_tensorly"),
                                                          nonnegative=False)
        print("(processed time: {:<.2f})".format(time.time() - f0))

    if(b15):
        f0 = time.time()
        tensorDecomp.tensorDecomp.doTensorDecomp_tensorly(TENSOR.joinpath("tensor3", "tensor.pickle"),
                                                          TENSOR.joinpath("tensor3", "words.dat"),
                                                          TENSOR.joinpath("tensor3", "mIDs.dat"),
                                                          RESULT.joinpath("p_r_tgtset_explan", "tensor", "tensor3", "TuckerDecomp_tensorly_nonnegative2",),
                                                          nonnegative=True)
        print("(processed time: {:<.2f})".format(time.time() - f0))


    # --------------------------------------------------------- MULTI CHANNEL ---------------------------------------------------------

    forder = MORPHOMES.joinpath("multi_channel")
    input_ = [MORPHOMES.joinpath("sompo_full003.json"), MORPHOMES.joinpath("sompo_full005.json"),
              MORPHOMES.joinpath("sompo1_10000.json"), MORPHOMES.joinpath("sompo2_10000.json"),
              MORPHOMES.joinpath("sompo1_30000.json"), MORPHOMES.joinpath("sompo2_30000.json"),
              MORPHOMES.joinpath("sompo_first_10000_1.json"), MORPHOMES.joinpath("sompo_first_10000_2.json")]
    morphomes = [forder.joinpath("multi_full003.json"), forder.joinpath("multi_full005.json"),
                 forder.joinpath("multi1_10000.json"), forder.joinpath("multi2_10000.json"),
                 forder.joinpath("multi1_30000.json"), forder.joinpath("multi2_30000.json"),
                 forder.joinpath("multi_first_10000_1.json"), forder.joinpath("multi_first_10000_2.json")]
    # input_ = [MORPHOMES.joinpath("sompo_first_10000_1.json"), MORPHOMES.joinpath("sompo_first_10000_2.json")]
    # morphomes = [forder.joinpath("multi_first_10000_1.json"), forder.joinpath("multi_first_10000_2.json")]
    tensors = [TENSOR.joinpath("multi_channel", a.stem + ".pickle") for a in morphomes]

    if(b16):
        f0 = time.time()
        for m in range(len(input_)):
            shutil.copyfile(input_[m], morphomes[m])
            for n in range(len(MC_TEXT_KEYS_FOR_TYPE)):
                print("")
                sepReviewsToMorphomes.sepReviews(morphomes[m],
                                                 morphomes[m],
                                                 inputTextKey=MC_TEXT_KEYS_FOR_TYPE[n],
                                                 outputTextKey=MC_TEXT_KEYS_FOR_TYPE[n] + "_morphomes",
                                                 blacklist=r"(名詞-数)", whitelist=r"(^名詞)",
                                                 blackWord=["～", "(", ")", "/", "~", "+", "-", ",", "－", "⇒",
                                                            "こと", "もの", "事", "よう", "ため", "の", "そう"],
                                                 #  removeRateFromBtm=(0.03 if m != 1 else 0.05),
                                                 removeRateFromBtm=0.03,
                                                 removeRateFromTop=0.15,
                                                 minWordsInSentence=0)
            print("(processed time: {:<.2f})".format(time.time() - f0))

    if(b17):
        f0 = time.time()
        for n in range(len(morphomes)):
            tensorDecomp.makeTensor.makeTensorForMultiChannel(morphomes[n], HEALTH_CHECK_REFINED, tensors[n],
                                                              MULTI_CHANNEL_KEYS,
                                                              HC_MEASUREMENT_KEYS, HC_HABIT_KEYS, HC_MEDICINE_KEYS)
            print("")
        print("(processed time: {:<.2f})".format(time.time() - f0))

    if(b18):
        f0 = time.time()
        LDA_pyro.excute.excuteLDAForMultiChannel("LDA", tensors[2], morphomes[2],
                                                 RESULT.joinpath("multi_channel", "LDA", "tmp"))
        print("(processed time: {:<.2f})".format(time.time() - f0))

    if(b19):
        f0 = time.time()
        LDA_pyro.excute.excuteLDAForMultiChannel("MCLDA", tensors[2], morphomes[2],
                                                 RESULT.joinpath("multi_channel", "MCLDA", "tmp"))
        print("(processed time: {:<.2f})".format(time.time() - f0))

    if(b22):
        f0 = time.time()
        LDA_pyro.excute.excuteLDAForMultiChannel("MCLDAnum", tensors[2], morphomes[2],
                                                 RESULT.joinpath("multi_channel", "MCLDAnum", "tmp"),
                                                 habit_levels=HC_HABIT_LEVELS, medicine_levels=HC_MEDICINE_LEVELS)
        print("(processed time: {:<.2f})".format(time.time() - f0))

    if(b23):
        f0 = time.time()
        LDA_pyro.excute.excuteLDAForMultiChannel("MCLDAnum_only", tensors[2], morphomes[2],
                                                 RESULT.joinpath("multi_channel", "MCLDAnum_only", "tmp"))
        print("(processed time: {:<.2f})".format(time.time() - f0))

    if(b25):
        f0 = time.time()
        # LDA_pytorch.excute.excuteMCLDA(morphomes[2], tensors[2],
        #                                RESULT.joinpath("multi_channel", "torch", "MCLDA", "K8 step200 seed1"),
        #                                pathTestdocs=morphomes[3], pathTesttensors=tensors[3])
        LDA_pytorch.excute.excuteMCLDA(morphomes[6], tensors[6],
                                       RESULT.joinpath("multi_channel", "torch", "MCLDA", "first K8 step200 seed1"),
                                       pathTestdocs=morphomes[7], pathTesttensors=tensors[7])
        # LDA_pytorch.excute.excuteMCLDA(morphomes[6], tensors[6],
        #                                RESULT.joinpath("multi_channel", "torch", "MCLDA", "tmp"),
        #                                pathTestdocs=morphomes[7], pathTesttensors=tensors[7])
        print("(processed time: {:<.2f})".format(time.time() - f0))

    if(b27):
        f0 = time.time()
        LDA_pytorch.excute.excuteMCLDA(morphomes[6], tensors[6],
                                       RESULT.joinpath("multi_channel", "torch", "MCLDA_direction", "tmp"),
                                       pathTestdocs=morphomes[7], pathTesttensors=tensors[7], forDerection=True)
        print("(processed time: {:<.2f})".format(time.time() - f0))

    if(b26):
        f0 = time.time()
        LDA_pytorch.excute.excuteMCLDA_K_range(morphomes[2], tensors[2],
                                               #    RESULT.joinpath("multi_channel", "torch", "MCLDA", "Ks"),
                                               Path("Ks1-20x10"),
                                               pathTestdocs=morphomes[3], pathTesttensors=tensors[3])
        print("(processed time: {:<.2f})".format(time.time() - f0))

    if(b28):
        f0 = time.time()
        analysis.cca.CCA_between_thetas(
            RESULT.joinpath("multi_channel", "torch", "MCLDA", "first K8 step200 seed1", "theta.csv"),
            # RESULT.joinpath("multi_channel", "torch", "MCLDA_direction", "K20 seed1", "theta.csv"),
            RESULT.joinpath("p_r_tgtset_explan", "sompo_19_first", "LLDAS30", "vectors.csv"),
            RESULT.joinpath("cca", "tmp")
        )
        print("(processed time: {:<.2f})".format(time.time() - f0))


    # --------------------------------------------------------- HEALTH CHECK ANALYSIS ---------------------------------------------------------

    if(b11):
        f0 = time.time()
        analysis.analyzeHealthCheck.analyze(
            HEALTH_CHECK, ANALYSIS_HEALTH_CHECK)
        print("(processed time: {:<.2f})".format(time.time() - f0))
