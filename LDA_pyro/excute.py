import pickle
import json

import numpy as np
import torch
import pyro
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
from pyro.infer import MCMC, NUTS
from pyro import distributions as dist
import matplotlib.pyplot as plt

import LDA_pyro.LDA as LDA
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


def excuteFromData(modelType, bow, words, docs, pathResult, counts=None):

    args = type("args", (object,), {})
    args.modelType = modelType
    args.device = DEVICE

    if(modelType == "LDA" or modelType == "LDA_auto"):
        args.num_steps = 200
        args.learning_rate = 0.2
        args.K = 7
        args.autoHyperParam = (modelType == "LDA_auto")
        args.coef_beta = 1
        args.coef_alpha = 1  # 大きいほうが同じようなトピックが形成される感じ
        args.eps = 0.001

        data = bow
        model = LDA.model
        guide = LDA.guide
        summary = LDA.summary
        if(not args.autoHyperParam):
            print(f"coef_beta:   {args.coef_beta}")
            print(f"coef_alpha: {args.coef_alpha}")
        else:
            print("Hyper parameter auto adapted")

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

    elif(modelType == "MCLDA" or modelType == "MCLDA_auto"):
        args.num_steps = 200
        args.learning_rate = 0.2
        args.K = 7
        args.autoHyperParam = (modelType == "MCLDA_auto")
        args.coef_beta = 1
        args.coef_alpha = 1  # 大きいほうが同じようなトピックが形成される感じ
        args.eps = 0.001

        data = bow
        model = MCLDA.model
        guide = MCLDA.guide
        summary = MCLDA.summary
        if(not args.autoHyperParam):
            print(f"coef_beta:   {args.coef_beta}")
            print(f"coef_alpha: {args.coef_alpha}")
        else:
            print("Hyper parameter auto adapted")

    # SVI
    svi = SVI(model=model,
              guide=guide,
              optim=Adam({"lr": args.learning_rate}),
              loss=Trace_ELBO())
    pyro.clear_param_store()
    losses = []
    for i in range(args.num_steps):
        loss = svi.step(data, args=args)
        losses.append(loss)
        if((i + 1) % 10 == 0):
            print("i:{:<5d} loss:{:<f}".format(i + 1, loss))

    pathResult.mkdir(exist_ok=True, parents=True)
    summary(data, args, words, docs, pathResult, counts=counts)

    plt.plot(losses)
    plt.savefig(pathResult.joinpath("loss.png"))
    plt.clf()

    print("")


def excuteLDAFromPath(modelType, pathTensor, pathWords, pathDocs, pathResult):

    with open(str(pathTensor), 'rb') as f:
        bow = pickle.load(f)
    with open(str(pathWords), 'r', encoding="utf_8_sig") as f:
        words = f.readline().split("\t")
    with open(str(pathDocs), "r", encoding="utf_8_sig") as f:
        docs = json.load(f)
    bow = torch.tensor(bow)

    excuteFromData(modelType, bow, words, docs, pathResult)


def excuteLDAForMultiChannel(modelType, pathTensors, pathDocs, pathResult):

    with open(str(pathTensors), 'rb') as f:
        tensors = pickle.load(f)
    with open(str(pathDocs), "r", encoding="utf_8_sig") as f:
        docs = json.load(f)

    keys = tensors["tensor_keys"]

    if(modelType == "LDA" or modelType == "LDA_auto"):
        for key in keys:
            bow = torch.tensor(tensors[key], device=DEVICE)
            words = tensors[key + "_words"]
            counts = tensors[key + "_counts"]
            excuteFromData(modelType, bow, words, docs, pathResult.joinpath(key), counts=counts)

    elif(modelType == "MCLDA" or modelType == "MCLDA_auto"):
        bows = []
        words = []
        counts = []
        for key in keys:
            bows.append(torch.tensor(tensors[key], device=DEVICE))
            words.append(tensors[key + "_words"])
            counts.append(tensors[key + "_counts"])
        excuteFromData(modelType, bows, words, docs, pathResult, counts=counts)
