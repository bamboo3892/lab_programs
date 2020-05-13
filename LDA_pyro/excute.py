import pickle
import json

import numpy as np
import torch
import pyro
from pyro.optim import Adam, SGD
from pyro.infer import SVI, Trace_ELBO
from pyro.infer import MCMC, NUTS
from pyro import distributions as dist
import matplotlib.pyplot as plt

import LDA_pyro.LDA as LDA
import LDA_pyro.LDA_original as LDA_original
import LDA_pyro.LDA_mcmc as LDA_mcmc
import LDA_pyro.LDA_original_collapsed as LDA_original_collapsed
import LDA_pyro.sepSW as sepSW
import LDA_pyro.sepSW_update_model as sepSW2
import LDA_pyro.sNTD as sNTD
import LDA_pyro.sNTD_update_model as sNTD2
import LDA_pyro.eLLDA as eLLDA
import LDA_pyro.MCLDA as MCLDA


seed = 1
pyro.set_rng_seed(seed)
np.random.seed(seed)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def excuteFromData(modelType, bow, words, docs, pathResult, counts=None, *,
                   args=type("args", (object,), {})):

    pathResult.mkdir(exist_ok=True, parents=True)
    inference = "svi"
    args.modelType = modelType
    args.device = DEVICE

    if(modelType == "LDA"):
        args.num_steps = 200
        args.learning_rate = 0.2
        args.K = 7
        args.auto_beta = True
        args.auto_alpha = True
        args.coef_beta = 1
        args.coef_alpha = 1  # 大きいほうが同じようなトピックが形成される感じ?
        args.eps = 0.00001

        data = bow
        model = LDA.model
        guide = LDA.guide
        summary = LDA.summary
        print(f"coef_beta:   {args.coef_beta} (auto: {args.auto_beta})")
        print(f"coef_alpha:  {args.coef_alpha} (auto: {args.auto_alpha})")

    if(modelType == "LDA_original"):
        words, data = _func0(docs, "morphomes", DEVICE)

        args.num_steps = 100
        args.learning_rate = 0.1
        args.D = (data[1][len(data[1]) - 1] + 1).cpu().detach().numpy().tolist()
        args.V = len(words)
        args.K = 7
        args.auto_beta = False
        args.auto_alpha = False
        args.coef_beta = 1
        args.coef_alpha = 1

        model = LDA_original.model
        guide = LDA_original.guide
        summary = LDA_original.summary

    if(modelType == "LDA_mcmc"):
        words, data = _func0(docs, "morphomes", DEVICE)

        args.jit = True
        args.D = (data[1][len(data[1]) - 1] + 1).cpu().detach().numpy().tolist()
        args.V = len(words)
        args.K = 7
        args.auto_beta = False
        args.auto_alpha = False
        args.coef_beta = 1
        args.coef_alpha = 1

        inference = "mcmc"
        model = LDA_mcmc.model
        summary = LDA_mcmc.summary

    elif(modelType == "sepSW"):
        seasons = torch.tensor([int(review["season"]) for review in docs])

        args.num_steps = 1
        args.learning_rate = 0.2
        args.D = bow.shape[0]
        args.K = 7
        args.V = bow.shape[1]
        args.S = 4
        args.coef_phi = 1
        args.coef_sigma = 1
        args.coef_theta = 11
        args.eps = 0.001
        print(f"coef_phi:   {args.coef_phi}")
        print(f"coef_sigma: {args.coef_sigma}")
        print(f"coef_theta: {args.coef_theta}")

        data = [bow, seasons]
        model = sepSW.model
        guide = sepSW.guide
        summary = sepSW.summary
        # model = sepSW2.model
        # guide = sepSW2.guide
        # summary = sepSW2.summary

    elif(modelType == "sNTD"):
        seasons = torch.tensor([int(review["season"]) for review in docs])

        args.num_steps = 200
        args.learning_rate = 0.2
        args.D = bow.shape[0]
        args.K1 = 10
        args.K2 = 4
        args.V = bow.shape[1]
        args.S = 4
        args.coef_phi = 0.01
        args.coef_sigma = 1
        args.coef_theta = 1
        args.eps = 0.001
        print(f"coef_phi:   {args.coef_phi}")
        print(f"coef_sigma: {args.coef_sigma}")
        print(f"coef_theta: {args.coef_theta}")

        data = [bow, seasons]
        model = sNTD.model
        guide = sNTD.guide
        summary = sNTD.summary
        # model = sNTD2.model
        # guide = sNTD2.guide
        # summary = sNTD2.summary

    elif(modelType == "eLLDA"):
        labels = [review["labels"] for review in docs]

        args.num_steps = 1
        args.learning_rate = 0.2
        args.D = bow.shape[0]
        args.K = 15
        args.V = bow.shape[1]
        args.coef_phi = 1
        args.coef_theta1 = 10  # その文書がそのラベル思っていない時
        args.coef_theta2 = 100  # その文書がそのラベル思っている時
        # args.coef_theta3 = 1  # ラベルなしトピック
        args.eps = 0.001

        # num_labels = len(np.unique(np.array(labels).flatten()))
        theta_prior = np.ones([args.D, args.K]) * args.coef_theta1
        for d in range(args.D):
            theta_prior[d, labels[d]] = args.coef_theta2
        theta_prior = torch.tensor(theta_prior)

        data = [bow, theta_prior]
        model = eLLDA.model
        guide = eLLDA.guide
        summary = eLLDA.summary

    elif(modelType == "MCLDA"):
        args.num_steps = 200
        args.learning_rate = 0.2
        args.K = 5
        args.autoHyperParam = True
        args.coef_beta = 1
        args.coef_alpha = 1  # 大きいほうが同じようなトピックが形成される感じ
        args.eps = 0.01

        data = bow
        model = MCLDA.model
        guide = MCLDA.guide
        summary = MCLDA.summary
        if(not args.autoHyperParam):
            print(f"coef_beta:   {args.coef_beta}")
            print(f"coef_alpha: {args.coef_alpha}")
        else:
            print("Hyper parameter auto adapted")

    if(inference == "svi"):
        # SVI
        svi = SVI(model=model,
                  guide=guide,
                  optim=Adam({"lr": args.learning_rate}),
                  #   optim=SGD({"lr": 0.1}),
                  loss=Trace_ELBO())
        pyro.clear_param_store()

        i = 0
        losses = []
        while(True):
            loss = svi.step(data, args=args)
            losses.append(loss)
            # if((i + 1) % 10 == 0):
            #     print("i:{:<5d} loss:{:<f}".format(i + 1, loss))
            # p = pathResult.joinpath(f"{i+1}")
            # p.mkdir(exist_ok=True, parents=True)
            # summary(data, args, words, docs, p, counts=counts)

            i += 1
            if(args.num_steps is not None):
                if(i >= args.num_steps):
                    break
            else:
                # TODO 収束判定
                break

        plt.plot(losses)
        plt.savefig(pathResult.joinpath("loss.png"))
        plt.clf()

    elif(inference == "mcmc"):
        nuts_kernel = NUTS(model, jit_compile=args.jit)
        mcmc = MCMC(nuts_kernel,
                    num_samples=1000,
                    # warmup_steps=args.warmup_steps,
                    num_chains=7
                    )
        mcmc.run(data, args)

    summary(data, args, words, docs, pathResult, counts=counts)

    print("")


def excuteLDAFromPath(modelType, pathTensor, pathWords, pathDocs, pathResult):

    with open(str(pathTensor), 'rb') as f:
        bow = pickle.load(f)
    with open(str(pathWords), 'r', encoding="utf_8_sig") as f:
        words = f.readline().split("\t")
    with open(str(pathDocs), "r", encoding="utf_8_sig") as f:
        docs = json.load(f)
    bow = torch.tensor(bow, device=DEVICE)

    excuteFromData(modelType, bow, words, docs, pathResult)

    # args = type("args", (object,), {})
    # for i in range(3):
    #     for j in range(3):
    #         args.coef_beta = [0.1, 1, 10][i]
    #         args.coef_alpha = [0.1, 1, 10][j]

    #         excuteFromData(modelType, bow, words, docs,
    #                        pathResult.joinpath(f"b{args.coef_beta}f_a{args.coef_alpha}f"),
    #                        args=args)


def excuteLDAForMultiChannel(modelType, pathTensors, pathDocs, pathResult):

    with open(str(pathTensors), 'rb') as f:
        tensors = pickle.load(f)
    with open(str(pathDocs), "r", encoding="utf_8_sig") as f:
        docs = json.load(f)

    keys = tensors["tensor_keys"]

    if(modelType == "LDA"):
        for key in keys:
            print(f"key: {key}")
            bow = torch.tensor(tensors[key], device=DEVICE)
            words = tensors[key + "_words"]
            counts = tensors[key + "_counts"]
            excuteFromData(modelType, bow, words, docs, pathResult.joinpath(key), counts=counts)
            return

    elif(modelType == "MCLDA"):
        bows = []
        words = []
        counts = []
        for key in keys:
            bows.append(torch.tensor(tensors[key], device=DEVICE))
            words.append(tensors[key + "_words"])
            counts.append(tensors[key + "_counts"])
        excuteFromData(modelType, bows, words, docs, pathResult, counts=counts)


def _func0(docs, morphomes_key, device):
    words = []
    data0 = []  # word id
    data1 = []  # doc id
    for d, doc in enumerate(docs):
        if(d >= 3000):
            break
        sentences = doc[morphomes_key]
        for sentence in sentences:
            for word in sentence:
                if(word not in words):
                    words.append(word)
                data0.append(words.index(word))
                data1.append(d)

    return words, [torch.tensor(data0, dtype=torch.int16, device=device),
                   torch.tensor(data1, dtype=torch.int16, device=device)]
